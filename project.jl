### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ a9548adc-cc71-4f26-96d2-7aefc39a554f
begin
	using DFTK
	using LinearAlgebra
	using Plots
	using PlutoTeachingTools
	using PlutoUI
	using Printf
	using BenchmarkTools
	using LinearMaps
	using DoubleFloats
	using Brillouin
	using Interpolations
end

# ╔═╡ 254bfb5d-8b0b-4cba-8bc4-c32c0312f690
#https://docs.sciml.ai/LinearSolve/stable/basics/Preconditioners/
begin
	using LimitedLDLFactorizations
end

# ╔═╡ 57dcd7d6-edd4-4c02-a3fd-dd74aeb5598a
begin
	using Preconditioners
end

# ╔═╡ 4b7ade30-8c59-11ee-188d-c3d4588f7106
md"Error control in scientific modelling (MATH 500, Herbst)"

# ╔═╡ 35066ab2-d52d-4a7f-9170-abbae6216996
md"""
# Semester project: Band structures with guaranteed error bars
*To be handed in via moodle by 20.12.2024*
"""

# ╔═╡ f09f563c-e458-480f-92be-355ab4582fb5
TableOfContents()

# ╔═╡ 3de7dfe4-032b-4e05-a929-5ffabfbb35cc
md"""
Hi my name is Bardia
"""

# ╔═╡ c17073d6-94e2-464f-acff-6f444bf7d08c
md"""
## Introduction

In this project we will compute approximations of
parts of the eigenspectra of a specific class of Hamiltonians
```math
H = -\frac12 \Delta + V
```
where $V$ are periodic potentials.

In the first part of the project we will combine some ideas from the course to come up with a tailored iterative eigensolver for such problems, which is as fast as possible.

In the second part we will develop mathematically guaranteed
estimates for the total numerical error of the eigenvalue computation, that is consider all error contributions from the discretisation, the algorithms and the floating-point error. As a result the estimates will reliably demonstrate that our exact answer is within the obtained error bounds."""

# ╔═╡ a4f9260b-7016-4a12-8fd1-230b8e234496
md"""
## Part 1: Eigensolver for finite-difference discretisations

We consider the Hamiltonian 
```math
	H_\text{per} = -\frac12 \Delta + V(x)
```
where $V(x) = \cos(x)$.
This potential is periodic and we thus consider the domain $\Omega = [0, 2\pi]$.

In this Part 1 we will only consider eigenfunctions of $H_\text{per}$, which are also periodic. It may be a little surprising, but this is a restriction as some parts of the spectrum of $H_\text{per}$ can be associated with eigenfunctions, that are *not* periodic. More details on this in the [lecture on periodic problems](https://teaching.matmat.org/error-control/11_Periodic_problems.html).

We employ a discretisation using finite differences, that is we split $\Omega$ into $N$ equal intervals with width $h = \frac{2π}{N}$, such that the grid points are $x_k = k \cdot h$. According to second-order finite differences
the Hamiltonian applied to a function $\psi$ can be expressed as
```math
\tag{1}
\big(H_\text{per} ψ\big)(x_k) \approx \frac 1 2 \frac{-ψ(x_{k-1}) + 2 ψ(x_k) - ψ(x_{k+1})}{h^2} + V(x_k) ψ(x_k),
```
for $k\in\{1,\dotsc,N\}$.
To also impose **periodic boundary conditions** on $\Omega = [0, 2\pi]$
for $\psi$ we use the convention $x_0 = x_N$ and $x_{N+1} = x_1$.

As we already noted for the Laplacian in earlier lectures,
the spectrum of a Hamiltonian like $-\frac12 \Delta + V(x)$ strongly depends on the imposed boundary conditions. To explore this a little we will also consider a closely related problem: we take a Hamiltonian with the same functional form $H_\text{dir}= -\frac12 \Delta + V(x)$ for the operator, but this time impose **homogeneous dirichlet boundary conditions** on the boundary of the domain $\Omega = [0, 2\pi]$, which corresponds to $\psi(0) = \psi(2\pi) = 0$. For the discretisation in equation (1) this means that we take $ψ(x_{0}) = ψ(x_{N+1}) = 0$.[^1]

In the following we will use $\textbf{H}_\text{per}$ to refer to the matrix resulting from the finite-difference discretisation of $H_\text{per}$
and similarly employ $\textbf{H}_\text{dir}$ to refer to the discretised version of $H_\text{dir}$. In Part 1 we will be mainly concerned with these matrices.

[^1]: For finite $N$, this specific discretisation formulation actually corresponds to enforcing the Dirichlet condition at $x_{N+1} = 2\pi + h$ instead of $2\pi$. Other formulations are possible, but not needed for the point of this exercise.
"""

# ╔═╡ 1bfb62aa-4b01-4839-8073-2a196162cd17
md"""
### Task 1.1: Finite-difference discretisation and familiarisation

**(a)** Implement a general function `fd_hamiltonian_periodic(V, N; T=Float64)` where `V` is a Julia function for representing the potential and $N$ is the number of finite-difference grid points. The function should return a `Symmetric` matrix representing the discretised Hamiltonian $\textbf{H}_\text{per}$ in **periodic boundary conditions**.  Make sure your function is type-stable with respect to the floating-point type `T`: if `T=Float32`, for example, all computation should be done in `Float32`.
Consider the case $V(x) = 0$ and compute the lowest 3 eigenvalues using `eigen` and the function you implemented. These should be around $0$ and two eigenvalues around $0.5$.

*Hint:* For (a) and the entire task 1.1. it will be sufficient to consider values of $N < 5000$. It is generally fine to use `eigen` to compute eigenpairs. We will consider building iterative approaches in Task 1.2, 1.3 and 1.4.
"""

# ╔═╡ d5c66274-47ec-41e3-a539-dc932ac7939a
function fd_hamiltonian_periodic(V, N; T=Float64)
	h = T(2π / N)
	x = T.(h * (1:N))

	H = SymTridiagonal(2ones(T, N), - ones(T, N-1)) / (2h^2) + diagm(V.(x))
	
	H[1, end] = -1 / 2h^2
	H[end, 1] = -1 / 2h^2

	return Symmetric(H)
end

# ╔═╡ 9b6218d6-b621-4ffd-b55a-fe3845357420
let
	res_fp64 = fd_hamiltonian_periodic(cos, 5; T=Float64)
	res_fp32 = fd_hamiltonian_periodic(cos, 5; T=Float32)
	ref_evals = [-0.563276, 0.0308699, 0.784539, 1.05227, 1.86188]

	if res_fp64 == Diagonal(ones(200))
		warning_box(md"Replace `Diagonal(ones(200))` by your answer")
	elseif maximum(abs, eigvals(res_fp64) .- ref_evals) > 1e-3
		keep_working()
	elseif !type_eq(res_fp64, Symmetric{Float64, Matrix{Float64}})
		almost(md"Almost there. Make sure to return a `Symmetric` matrix instead of a simple `Matrix`. Otherwise the numerics will be too challenging.")
	elseif !type_eq(res_fp32, Symmetric{Float32, Matrix{Float32}})
		almost(md"Almost there. Only the types are not yet quite right when T=Float32")
	end
end

# ╔═╡ 2689ed80-d936-4044-943b-b198cf57cdd1
# Compute lowest 3 eigenvalues for V(x) = 0
begin
	zero(x) = 0
	eigen(fd_hamiltonian_periodic(zero, 1000)).values[1:3]
end

# ╔═╡ 8c8e8409-4e8a-4e29-875c-dbc969a3634d
md"""
**(b)**  Copy your code to a second function `fd_hamiltonian_dirichlet` computing $\textbf{H}_\text{dir}$. For this remove the condition for periodicity, i.e. employ **Dirichlet boundary conditions** by setting $\psi(x_0) = \psi(x_{N+1}) = 0$. Keep in mind that Julia features types to exploit the special structure of this matrix, which we used in previous lectures and exercises. Consider its lowest three eigenvalues. What changes ?
"""

# ╔═╡ 0e3c162b-bfbc-42e0-986c-aebb328cc2ed
function fd_hamiltonian_dirichlet(V, N; T=Float64)
	h = T(2π / N)
	x = T.(h * (1:N))
	H = SymTridiagonal(2ones(T, N), - ones(T, N-1)) / (2h^2) + diagm(V.(x))
	return SymTridiagonal(H)
end

# ╔═╡ 347744bb-8670-4d67-855d-982ae6c35f86
let
	res_fp64 = fd_hamiltonian_dirichlet(cos, 5; T=Float64)
	res_fp32 = fd_hamiltonian_dirichlet(cos, 5; T=Float32)
	ref = SymTridiagonal([0.942274, -0.17576, -0.17576, 0.942274, 1.63326],
		                 -0.316629ones(4))

	if res_fp64 == Diagonal(ones(200))
		warning_box(md"Replace `Diagonal(ones(200))` by your answer")
	elseif maximum(abs, ref .- res_fp64) > 1e-3
		keep_working()
	elseif !type_eq(res_fp64, SymTridiagonal{Float64, Vector{Float64}})
		almost(md"Almost there. Make sure you use a type that is more appropriate for the matrix structure. Otherwise the numerics will be too challenging.")
	elseif !type_eq(res_fp32, SymTridiagonal{Float32, Vector{Float32}})
		almost(md"Almost there. Only the types are not yet quite right when T=Float32")
	end
end

# ╔═╡ 27df213e-7e3e-4d25-880b-d0cce3563c04
# Lowest 3 eigenvalues
eigen(fd_hamiltonian_dirichlet(zero, 1000)).values[1:3]

# ╔═╡ 50cc1f78-4de0-48f4-876a-d30b4ef3d8d0
md"""
**(c)** We now turn our attention to the case $V(x) = \cos(x)$ for the rest of part 1. Using your functions above, compute again the lowest four eigenpairs for both boundary conditions. Plot in each case the corresponding eigenfunctions. What do you notice ?
"""

# ╔═╡ 915c79fc-5d79-4d61-acac-5e4b9618814d
function standardize_eigenvectors(eigenvectors, tol=-1e-6)
    for i in 1:size(eigenvectors, 2)
        if eigenvectors[1, i] < tol
            eigenvectors[:, i] .= -eigenvectors[:, i]
        end
    end
    return eigenvectors
end

# ╔═╡ cfdbc7d7-21dd-4639-9a55-f9aa11ce746e
function plot_eigenfunctions(H, V, N, num)
	x = range(0, stop=2π, length=N)
	evs = eigen(H(V, N))
	plot()
	for i in range(1, num)
		plot!(x, standardize_eigenvectors(evs.vectors[:, i]), label=round(evs.values[i], sigdigits=3))
	end
	if H(V, N)[1, end] != 0
		title!("Plot of Eigenfunctions (Periodic boundary)")
	else
		title!("Plot of Eigenfunctions (Dirichlet boundary)")
	end
end

# ╔═╡ 099a4340-bb0b-4eef-a630-ea94fba4d70c
plot_eigenfunctions(fd_hamiltonian_periodic, cos, 1000, 4)

# ╔═╡ 9dd3d49a-ac5a-472e-aec6-d758ab604e08
plot_eigenfunctions(fd_hamiltonian_dirichlet, cos, 1000, 4)

# ╔═╡ cb07dfa3-5151-4993-a8b4-62190ccace02
md"""
**(d)** Staying with $V(x) = \cos(x)$ consider the convergence of the first four eigenpairs of the Hamiltonian with $N$. What do you notice ?

*Optional as beyond the scope of our course:* Can you explain why ?
"""


# ╔═╡ 5432afd8-3e00-4756-ab89-d0315625330d
begin
	# Caclulate Exact Eigenspectrum (N=5000)
	eigenspectrum_exact_dir = eigen(fd_hamiltonian_dirichlet(cos, 5000))
	eigenspectrum_exact_per = eigen(fd_hamiltonian_periodic(cos, 5000))
end

# ╔═╡ 6903d208-e9df-4a3e-b938-95ebee57ea90
function interpolate_function(f, N=5000)
	x = range(2π/length(f), stop=2π, length=length(f))
	itp = cubic_spline_interpolation(x, f, extrapolation_bc=Line())
	x_fine = range(2π/N, stop=2π, length=N)
	fine_values = itp.(x_fine)
	return fine_values
end

# ╔═╡ f32f840c-c8f4-437a-8ac9-5b6cba2abb34
function plot_convergence_with_N(model, eigenspectrum, num_N=100, num_evs=4)
	V = cos
	if model == "periodic"
		H = fd_hamiltonian_periodic
	elseif model == "dirichlet"
		H = fd_hamiltonian_dirichlet
	end
	
	error_eigenvalues = zeros(num_N, num_evs)
	error_eigenvectors = zeros(num_N, num_evs)
	eigenvalues_exact = eigenspectrum.values[1:num_evs]
	eigenvectors_exact = standardize_eigenvectors(eigenspectrum.vectors)
	test_N = range(10, 500, num_N)
	
	for (i, N) in enumerate(test_N)
		eigenspectrum_N = eigen(H(V, Int(floor(N))))
		error_eigenvalues[i, :] = abs.(eigenvalues_exact - eigenspectrum_N.values[1:num_evs])
		
		eigenvectors_N = standardize_eigenvectors(eigenspectrum_N.vectors)
		for j in 1:num_evs
			f_interpolated = interpolate_function(eigenvectors_N[:, j])
			f_exact = eigenvectors_exact[:, j]
			error_eigenvectors[i,j] = norm(f_exact - f_interpolated) * (2π / N)^(0.5)
		end
	end

	# Make layout and set up plots
	layout = @layout [a; b]
	plot(size=(800, 1000), layout=layout)

	# Add the first subplot
	plot!(yaxis=:log, subplot=1, label="Plot 1")
	
	# Add the second subplot
	plot!(yaxis=:log, subplot=2, label="Plot 2")
	
	for i in range(1, num_evs)
		plot!(test_N, error_eigenvalues[:, i], label="Eigenvalue #$i", subplot=1)
		plot!(test_N, error_eigenvectors[:, i], label="Eigenvector #$i", subplot=2)
	end

	xlabel!("N")
	title!("Convergence of Eigenvalues ($model boundary)", subplot=1)
	title!("Convergence of Eigenvectors ($model boundary)", subplot=2)
end
	

# ╔═╡ 86693da0-9b06-4c13-9d73-55ee364de1d8
plot_convergence_with_N("dirichlet", eigenspectrum_exact_dir)

# ╔═╡ 0a6c99c7-11c8-46f4-8ec2-6fb2166579f7
plot_convergence_with_N("periodic", eigenspectrum_exact_per)

# ╔═╡ 3d5e051f-13a9-415e-80bd-78cafc16b97a
md"""
**(e)** Assume we are able to compute the eigenpairs of the discretised Hamiltonian $\textbf{H}_\text{dir}$ using Dirichlet boundary conditions, but unable to do so for $\textbf{H}_\text{per}$. If you are curious why this might be a problem have a look at Task 1.2(a). 

Propose an approach using first-order perturbation theory to estimate the first $3$ eigenvalues and eigenvectors of $\textbf{H}_\text{per}$ using only the information you have on the eigenspectrum of $\textbf{H}_\text{dir}$.
For this write down on paper the situation for $N=4$ and inspect what changes between the discretised versions of $\textbf{H}_\text{dir}$ and $\textbf{H}_\text{per}$. Then propose an algorithm to compute the first-order perturbation correction in moving from $\textbf{H}_\text{dir}$ to $\textbf{H}_\text{per}$ in the discretised setting for general $N$.

Perform some numerical experiments.
How accurate is your perturbation-based approach compared to computing the actual eigenpairs of $\textbf{H}_\text{per}$ for various $N$. How does this accuracy depend on $N$ ?
"""

# ╔═╡ da1c7f17-aaf8-464a-b7b5-8ce2f2c5a1c4
function get_approx_dir_per_eigen(N, num_evs=3)
	# Assume we know eigenspectrum of H_dir
	H_dir = fd_hamiltonian_dirichlet(cos, N)
	eigenspectrum_H_dir = eigen(H_dir)
	ΔA = zeros(N, N)
	ΔA[1, end] = -1 / 2(2π / N)^2
	ΔA[end, 1] = -1 / 2(2π / N)^2

	x_approx = zeros(N, num_evs)
	λ_approx = zeros(num_evs)
	for i in 1:num_evs
		λ_dir = eigenspectrum_H_dir.values[i]
		x_dir = standardize_eigenvectors(eigenspectrum_H_dir.vectors)[:, i]
		
		Δλ = dot(x_dir, ΔA * x_dir)
	
		Q_λ = I - x_dir * x_dir' 
		Δx = - (H_dir - λ_dir * I) \ (Q_λ * ΔA * x_dir)
	
		x_per = x_dir + Δx
		λ_per = λ_dir + Δλ

		x_approx[:, i] = x_per
		λ_approx[i] = λ_per
	end

	λ_dirichlet = eigenspectrum_H_dir.values[1:num_evs]
	x_dirichlet = standardize_eigenvectors(eigenspectrum_H_dir.vectors)[:, 1:num_evs]

	H_per = fd_hamiltonian_periodic(cos, N)
	eigenspectrum_H_per = eigen(H_per)
	λ_periodic = eigenspectrum_H_per.values[1:num_evs]
	x_periodic = standardize_eigenvectors(eigenspectrum_H_per.vectors)[:, 1:num_evs]
	
	return (λ_approx, x_approx, λ_dirichlet, x_dirichlet, λ_periodic, x_periodic)
end

# ╔═╡ d870beac-018e-44c6-8356-8ba4e337c2cd
begin
	max_N = 500
	num_evs = 3
	test_N = 5:10:max_N
	errors_eigenvector = zeros(length(test_N), num_evs)
	errors_eigenvalue = zeros(length(test_N), num_evs)
	for (i,N) in enumerate(test_N)
		λ_approx, x_approx, λ_dirichlet, x_dirichlet, λ_periodic, x_periodic = get_approx_dir_per_eigen(N, num_evs)
		errors_eigenvector[i,:] = mapslices(norm, x_approx - x_periodic, dims=1) * (2π / N)^(0.5)
		errors_eigenvalue[i, :] = abs.(λ_approx - λ_periodic)
	end

	# Make layout and set up plots
	layout = @layout [a; b]
	plot(size=(800, 1000), layout=layout)

	# Add the first subplot
	plot!(yaxis=:log, subplot=1, label="Plot 1")
	
	# Add the second subplot
	plot!(yaxis=:log, subplot=2, label="Plot 2")
	
	plot!(test_N, errors_eigenvector, label=["Eigenvector #1" "Eigenvector #2" "Eigenvector #3"], subplot=1)

	plot!(test_N, errors_eigenvalue, label=["Eigenvalue #1" "Eigenvalue #2" "Eigenvalue #3"], subplot=2)
	
	title!("Error in pertubation theory approx eigenvector", subplot=1)
	title!("Error in pertubation theory approx eigenvalue", subplot=2)
end

# ╔═╡ eab701f2-b43e-4805-8a78-ee62dde5fc15
md"""
**(f)** We stay in the setting of (e) where the eigenpairs of $\textbf{H}_\text{dir}$ are accessible but the ones of $\textbf{H}_\text{per}$ not. We want to develop a computable error bound, which estimates the error we make when employing the eigenvalues of $\textbf{H}_\text{dir}$ as a stand-in for the eigenvalues of eigenpairs of $\textbf{H}_\text{per}$. Propose both 
- a guaranteed error bound and
- an error bound which is as sharp as possible (i.e. where the estimated error is close to the true error against the actual eigenvalues of $\textbf{H}_\text{per}$)
Demonstrate the accuracy of your bounds across increasing values for $N$.

At which discretisation parameter $N$ is your estimated model error (from using $\textbf{H}_\text{dir}$ instead of $\textbf{H}_\text{per}$) roughly on the same order as the discretisation error for each of the first three eigenvalues ?
"""

# ╔═╡ d32943b3-d098-4b9b-bab0-5b1871d9d5ac
md"""
- Bauer-Fike bound is a guaranteed error bound
- Symmetric Kato Temple bound with $\delta_{\text{est}}$ is sharp but not guaranteed
- The estimated model error is roughly on the same order as the discretisation error: 1) never for eigenvalue 1; 2) N < 500 for eigenvalue 2; and 3) N < 100 for eigenvalue 3
"""

# ╔═╡ 8d20abfb-63ff-424b-a560-cbad23ad1e84
begin
	bauer_fike_errors = zeros(length(test_N), num_evs)
	kato_temple_errors = zeros(length(test_N), num_evs)
	true_errors = zeros(length(test_N), num_evs)
	for (i,N) in enumerate(test_N)
		H_per = fd_hamiltonian_periodic(cos, N)
		_, _, λ_dirichlet, x_dirichlet, λ_periodic, x_periodic = get_approx_dir_per_eigen(N, num_evs+1)
		h = 1 / 2(2π / N)^2
		
		norm_r = h * sqrt.(x_dirichlet[1, :].^2 + x_dirichlet[end,:].^2)[1:num_evs]
		bauer_fike_errors[i, :] = norm_r
		true_errors[i, :] = abs.(λ_dirichlet - λ_periodic)[1:num_evs]
		
		for j in 1:num_evs
			if j == 1
				δ_est = abs(λ_dirichlet[j+1] - λ_dirichlet[j])
			else
				δ_est = min(abs(λ_dirichlet[j-1] - λ_dirichlet[j]), abs(λ_dirichlet[j+1] - λ_dirichlet[j]))
			end
			kato_temple_errors[i, j] = norm_r[j]^2 / δ_est
		end
	end

	layout_bound = @layout [a; b; c]
	p = plot(size=(800, 1000), layout=layout_bound)

	# Add the subplots
	plot!(yaxis=:log, subplot=1, label="Plot 1")
	plot!(yaxis=:log, subplot=2, label="Plot 2")
	plot!(yaxis=:log, subplot=3, label="Plot 2")

	for i in 1:3
		plot!(test_N, bauer_fike_errors[:, i], label="Bauer-Fike Error", subplot=i)
		plot!(test_N, kato_temple_errors[:, i], label="Kato Temple Error", subplot=i)
		plot!(test_N, true_errors[:, i], label="True Error", subplot=i)

		title!("Eigenvalue $i", subplot=i)
	end
	p
end

# ╔═╡ 58e69281-e2d7-40d9-9c1e-4b6807b405b7
md"""
------------------------
"""

# ╔═╡ cd99bdc8-dabf-4ddc-82a7-e79cd0f2e52b
md"""
### Task 1.2: A tailored iterative eigensolver

In task 1.1 we used `eigen` to compute the few lowest eigenpairs, now we will build up an iterative diagonalisation approach for such Hamiltonians.

**(a)** The following code performs a benchmark of the matrix returned by `fd_hamiltonian_periodic` for $N = 500$:
"""

# ╔═╡ 6002d1e6-57b6-4284-a773-409308e4b58f
let
	H = fd_hamiltonian_periodic(cos, 500);
	x = randn(size(H, 2))
	@btime $H * $x
end

# ╔═╡ f7343721-e40b-437d-871e-b7b52201d861
md"""
Perform the same benchmarks for the `\` (backslash operator, `H \ x`) with a random vector. Repeat with a factorised form of the Hamiltonian (`factorize(H)`). Also repeat all three benchmarks when you take `H` to be the result from `fd_hamiltonian_dirichlet`. Try also larger values like $N = 1000$ or $N = 4000$.

What do you observe? Comment on the balance of timings between `*` and `\` (in both factorised and unfactorised form) and the balance of timings between employing $\textbf{H}_\text{dir}$ and $\textbf{H}_\text{per}$. What is the advantage of factorisation ?
What is the advantage of the finite-difference Dirichlet Hamiltonian $\textbf{H}_\text{dir}$ over the finite-difference periodic Hamiltonian $\textbf{H}_\text{per}$ ? Coming back to Task 1.1 (e) and (f): Can you imagine why one would solve the eigenpairs of $\textbf{H}_\text{dir}$ even though one may actually be interested in periodic problems, thus $\textbf{H}_\text{per}$ ?

*Hint:* It might be helpful to read up on `@btime` and `@benchmark` to understand the syntax of `@btime`, especially the implications of the `$`.
"""

# ╔═╡ 89dc90a0-6c2c-4076-86b2-7ed79eccc988
#H_per\x , N = 500
let
	H = fd_hamiltonian_periodic(cos, 500);
	x = randn(size(H, 2));
	@btime $H \ $x
end

# ╔═╡ 8a5692e8-ebf4-4521-9d23-aff2bb0c4921
#factorized(H_per)\x , N = 500
let
	H = fd_hamiltonian_periodic(cos, 500);
	H = factorize(H);
	x = randn(size(H, 2));
	@btime $H \ $x
end

# ╔═╡ b68f0dce-89a4-4a9c-8f2e-4611895063df
#H_dir*x , N = 500
let
	H = fd_hamiltonian_dirichlet(cos, 500);
	x = randn(size(H, 2));
	@btime $H * $x
end

# ╔═╡ 03f6cbbf-f68f-40ed-9580-984ffa36f4e2
#H_dir\x , N = 500
let
	H = fd_hamiltonian_dirichlet(cos, 500);
	x = randn(size(H, 2));
	@btime $H \ $x
end

# ╔═╡ e4a34755-7a66-4bf8-bc90-7751188c7300
#factorized(H_dir)\x , N = 500
let
	H = fd_hamiltonian_dirichlet(cos, 500);
	H = factorize(H);
	x = randn(size(H, 2));
	@btime $H \ $x
end

# ╔═╡ ba2590e6-c9f6-46ff-84e0-20ecef148338
#factorized(H_dir)\x , N = 1000
let
	H = fd_hamiltonian_dirichlet(cos, 1000);
	H = factorize(H);
	x = randn(size(H, 2));
	@btime $H \ $x
end

# ╔═╡ 9127d505-3d0b-4ad0-b08e-ce441a74991e
#factorized(H_dir)\x , N = 4000
let
	H = fd_hamiltonian_dirichlet(cos, 4000);
	H = factorize(H);
	x = randn(size(H, 2));
	@btime $H \ $x
end

# ╔═╡ 772542e0-4676-46dc-827b-51dbeaa7f942
md"---"

# ╔═╡ ff257f67-2571-408a-88de-6afe220d2a67
md"""
Below a copy of the `lobpcg` routine of the lectures is defined.
"""

# ╔═╡ 3a0175c8-636e-40cd-963e-f7d9b30452a0
md"""
Notice, that instead of the QR-based orthogonalisation we considered in the lectures
(`ortho_qr`), this function uses an orthogonalisation based on Cholesky factorisation from the DFTK package (`ortho_cholesky`), which is generally faster. We will employ the DFTK package also in Part 2 below.
"""

# ╔═╡ 193a4535-efdc-4697-b734-dbefbd224b19
ortho_cholesky(X) = DFTK.ortho!(copy(X)).X

# ╔═╡ 727cb564-bf00-431a-a4d8-26d5fa2beddc
function lobpcg(A; X=randn(eltype(A), size(A, 2), 2), ortho=ortho_cholesky,
                Pinv=I, tol=1e-6, maxiter=100, verbose=true)
	T = real(eltype(A))
	m = size(X, 2)  # block size

	eigenvalues    = Vector{T}[]
	residual_norms = Vector{T}[]
	λ = NaN
	P = nothing
	R = nothing
	
	for i in 1:maxiter	
		if i > 1
			Z = hcat(X, P, R)
		else
			Z = X
		end
		Z = ortho(Z)
		AZ = A * Z
		λ, Y = eigen(Hermitian(Z' * AZ))
		λ = λ[1:m]
		Y = Y[:, 1:m]
		new_X = Z * Y

		R = AZ * Y - new_X * Diagonal(λ)
		norm_r = norm.(eachcol(R))
		push!(eigenvalues, λ)
		push!(residual_norms, norm_r)
		verbose && @printf "%3i %8.4g %8.4g\n" i λ[end] norm_r[end]
		if maximum(norm_r) < tol
			X = new_X
			break
		end

		R = Matrix(Pinv * R)
		P = X - new_X
		X = new_X
	end

	(; λ, X, eigenvalues, residual_norms)
end

# ╔═╡ b2f3187e-c468-493b-8d73-e70fb8459e85
ortho_qr(X) = Matrix(qr(X).Q)

# ╔═╡ 1026b80b-2f02-4a2c-ab79-3875e59a6836
let
	X = randn(3000, 10);
	@btime ortho_qr($X)
end;

# ╔═╡ c9cc14a6-4054-4fde-8b7e-abc0f4f8ffa4
let
	X = randn(3000, 10);
	@btime ortho_cholesky($X)
end;

# ╔═╡ bb62070f-4277-4ab5-9a8c-1f865c07aeab
md"""
**(b)** Use this `lobpcg` function to find the 3 smallest eigenpairs of the discretised Hamiltonian of $\textbf{H}_\text{per}$. Use $N = 500$ and converge until `tol = 1e-6`. You should find that a good preconditioner is crucial to get this problem to converge within 10--20 iterations. In order to find one, experiment a bit with the preconditioners available to you: take a look at the [lecture on diagonalisation routines](https://teaching.matmat.org/error-control/06_Diagonalisation_algorithms.html) to get some inspiration, but also keep in mind your findings in tasks 1.2(a) and 1.1(c). Try to find a setting which minimises the number of LOBPCG iterations. Increase $N$ beyond $500$ (e.g. consider $1000$ and $2000$) to ensure your selected preconditioner is stably better than some alternatives. 

*Hints:*
- A good way for finding a preconditioner is to test different preconditioners starting from the same initial guess, i.e. by making sure that for the different diagonalisations the iterations start from the identical intial guess `X`. To achieve this define yourself a random initial guess matrix `Xinit` and pass this matrix to all `lobpcg` runs as a keyword argument `X=Xinit`, e.g.
  ```julia
  Xinit = randn(500, 3)
  lobpcg(Hper; Pinv=first_preconditioner, X=Xinit, tol=1e-6)
  lobpcg(Hper; Pinv=second_preconditioner, X=Xinit, tol=1e-6)
  ```
- Look up how the `InverseMap` works.
"""

# ╔═╡ 7270fb1f-12eb-4a1a-a578-e3fc068cf698
#https://discourse.julialang.org/t/ldlt-factorization-for-full-matrices/75356
#https://docs.sciml.ai/LinearSolve/stable/basics/Preconditioners/
let
	Xinit = randn(500, 3)
	H = fd_hamiltonian_periodic(cos, 500);
	P1 = I #Identity Preconditioner
	P2 = Diagonal(1 ./ diag(H)) # Diagonal (Jacobi) Pinv=Diagonal(1 ./ diag(Apgd))
	#P3 = CholeskyPreconditioner(H, 2) #Not working
	#P4 = AMGPreconditioner{RugeStuben}(H) #Not working
	#P5 = AMGPreconditioner{SmoothedAggregation}(H) #Not working
	P6 = abs.(inv(lu(H).L)) #LUL Preconditioner (L)
	P7 = abs.(inv(lu(H).U)) #LUL Preconditioner (U)
	P8 = InverseMap(factorize(H)) #InverseMap Preconditioner
	print("Identity Preconditioner: \n")
	lobpcg(H; Pinv=P1, X=Xinit, tol=1e-6)
	print("Jacobi Preconditioner: \n ")
	lobpcg(H; Pinv=P2, X=Xinit, tol=1e-6)
	print("L Preconditioner: \n ")
	lobpcg(H; Pinv=P6, X=Xinit, tol=1e-6)
	print("U Preconditioner: \n ")
	lobpcg(H; Pinv=P7, X=Xinit, tol=1e-6)
	print("InverseMap Preconditioner: \n ")
	lobpcg(H; Pinv=P8, X=Xinit, tol=1e-6)
end

# ╔═╡ 00720b56-b681-4ff2-bc42-4d19318c7f74
md"""
**(c)** What matters most when designing a good preconditioner is that the *total runtime* to get the desired eigenpairs is minimised. This means that sometimes using more iteration, but employing less time *per iteration* can actually be better.

Taking a look at the `lobpcg` implementation above, explain how the choice of the preconditioner can influence the runtime. Keep in mind task 1.2(a) to suggest a preconditioner, which may need a few more LOBPCG iterations, but reduces the total runtime. Here, use $N=2000$ and employ the `@btime` macro on the `lobpcg` function call to measure the runtime in your experiments. This can be achieved for example like this:
```julia
res = @btime lobpcg($Hper; Pinv=$Pinv, X=$Xinit, tol=1e-6, verbose=false)
```
"""

# ╔═╡ 02ce43cd-9c83-4113-b326-852cf82c1751
#Check the time per iteration
let
	Xinit = randn(500, 3)
	Hper = fd_hamiltonian_periodic(cos, 500);
	P1 = I 
	P2 = Diagonal(1 ./ diag(Hper))
	P8 = InverseMap(factorize(Hper))
	print("Identity Preconditioner\n")
	res1 = @btime lobpcg($Hper; Pinv=$P1, X=$Xinit, tol=1e-6, maxiter=2000, verbose=false)
	print("Number of iterations:$(size(res1.eigenvalues,1))\n")

	print("Diagonal Preconditioner\n")
	res2 = @btime lobpcg($Hper; Pinv=$P2, X=$Xinit, tol=1e-6, maxiter=2000, verbose=false)
	print("Number of iterations:$(size(res2.eigenvalues,1))\n")
	
	print("InverseMap Preconditioner\n")
	res3 = @btime lobpcg($Hper; Pinv=$P8, X=$Xinit, tol=1e-6, maxiter=2000, verbose=false)
	print("Number of iterations:$(size(res3.eigenvalues,1))\n")
end

# ╔═╡ 3652b972-275c-424f-bee1-9f54b7ccd898
let
	Xinit = randn(2000, 3)
	Hper = fd_hamiltonian_periodic(cos, 2000);
	P8 = InverseMap(factorize(Hper))
	print("InverseMap Preconditioner\n")
	res3 = @btime lobpcg($Hper; Pinv=$P8, X=$Xinit, tol=1e-6, maxiter=2000, verbose=false)
	print("Number of iterations:$(size(res3.eigenvalues,1))\n")
end

# ╔═╡ a7eaade6-323d-4373-96e3-34dc81b3803e
md"""
### Results of Task 1.2
The Dirichlet Hamiltonian has a shorter computing time compared to the Periodic Hamiltonian.
Increasing the number of N can increase the computation time.
Using '\' instead of inv() can decrease the runtime.
Among all the preconditioners, the InverseMap has the best performance with the lowest total run time and the number of iterations; however, this preconditioner is costly and the time per iteration is longer than the identity or diagonal preconditioner and the lower iteration to reach the answer makes this preconditioner a great candidate with lowest total time.
"""

# ╔═╡ e5af416d-20a9-4812-99ff-45d62d6ab3a6
md"-----------------"

# ╔═╡ 66758cad-6aa8-496c-b2a6-3b1ca7c1fe42
md"""
### Task 1.3: Low-precision initial guess

Next to preconditioning we will now consider another trick to influence the runtime of diagonalisation algorithms.

Lower precision generally leads to faster computations. Higher precision generally leads to more accurate results. A natural idea is thus to chain computations at multiple precisions to get the best out of both worlds. 

We will work on the periodic Hamiltonian $H_\text{per}$, which can be generated for different precisions and $N=4000$ using the following function:
"""

# ╔═╡ d6930f90-2ded-4491-be63-582bfdb3b8d8
Htest(T=Float64) = fd_hamiltonian_periodic(cos, 4000; T);

# ╔═╡ 23afdfdd-885d-4aaa-baa0-78948b8fca48
md"""
**(a)** Using the `lobpcg` routine and the factorisation of $H_\text{per}$ as the preconditioner, determine roughly the highest accuracy (smallest residual norm) to which the lowest three eigenpairs of `Htest(T)` can be determined if `Float32`, `Float64` and `Double64` are used as the floating-point precision. Since the outcome is strongly dependent on the problem and setup, only an order of magnitude estimate is required here. The best way to obtain this is to repeat your experiments a few times. For `Double64` there is no need to go below `tol=1e-25`. 

In each run use the *the same* initial guess for each floating-point precision `T` to make the residual history comparable. However, ensure to convert this guess to the same floating-point precision before passing it to  `lobpcg` (e.g. `lobpcg(...; X=Float32.(X), ...)`). In one plot, show the residual norm history of the largest of the three eigenpairs (`last.(lobpcg( ... ).residual_norms)`) depending on the chosen floating-point precision `T`.

From your experiments: Roughly at which residual norm is it advisable to switch from one precision to the other in order to avoid impacting the rate of convergence ?
"""

# ╔═╡ 376a8e8e-c0be-413f-8035-dea8cd14f752
function lobpcg_new(A; X=randn(eltype(A), size(A, 2), 2), ortho=ortho_cholesky,
                Pinv=I, tol=1e-6, maxiter=100, verbose=true)
	T = real(eltype(A))
	T_pinv = eltype(Pinv)
	m = size(X, 2)  # block size

	eigenvalues    = Vector{T}[]
	residual_norms = Vector{T}[]
	λ = NaN
	P = nothing
	R = nothing
	
	for i in 1:maxiter	
		if i > 1
			Z = hcat(X, P, R)
		else
			Z = X
		end
		Z = ortho(Z)
		AZ = A * Z
		λ, Y = eigen(Hermitian(Z' * AZ))
		λ = λ[1:m]
		Y = Y[:, 1:m]
		new_X = Z * Y

		R = AZ * Y - new_X * Diagonal(λ)
		norm_r = norm.(eachcol(R))
		push!(eigenvalues, λ)
		push!(residual_norms, norm_r)
		verbose && @printf "%3i %8.4g %8.4g\n" i λ[end] norm_r[end]
		if maximum(norm_r) < tol
			X = new_X
			break
		end

		R = T_pinv.(R)
		R = Matrix(Pinv * R)
		P = X - new_X
		X = new_X
	end

	(; λ, X, eigenvalues, residual_norms)
end

# ╔═╡ e5bd8031-2c4a-4a62-b869-2ca4da4678ef
let H = Htest(Double64), X=randn(eltype(H), size(H, 2), 3), Pinv=InverseMap(factorize(Float32.(H))), niter = 50;
	
	#Float32
	result32=last.(lobpcg_new(Float32.(H), X=Float32.(X), maxiter=niter, Pinv=Pinv, tol=1e-25, verbose=false).residual_norms)
	plot(result32, yscale=:log10, label="Float32")
	
	#Float64
	result64=last.(lobpcg_new(Float64.(H), X=Float64.(X), maxiter=niter, Pinv=Pinv, tol=1e-25, verbose=false).residual_norms)
	plot!(result64, yscale=:log10, label="Float64")
	
	#Double64
	resultD64=last.(lobpcg_new(Double64.(H), X=Double64.(X), maxiter=niter, Pinv=Pinv, tol=1e-25, verbose=false).residual_norms)
	plot!(resultD64, yscale=:log10, label="Double64")

	xlabel!("iteration number")
	ylabel!("residual norm")
	title!("residual norm evolution with different types")
end

# ╔═╡ f2956e16-2e09-4c4e-a663-b8c5acb7b089
md"""
**(b)** With this outcome in mind code up a routine `solve_discretised(V, N; n_ep=3, tol32=XXX, tol=1e-6, maxiter=100)`, which first constructs the Hamiltonian using `fd_hamiltonian_periodic(V, Nb, a; T=Float32)` and then uses a preconditioned `lobpcg` to solve it for `n_ep` eigenpairs until `tol32` is reached, starting from a random guess. Then it switches to `Float64` as the working precision and continues the iterations by invoking `lobpcg` a second time. Note that for this you will need to recompute the Hamiltonian in `Float64`. Make sure the named tuple returned by the second call to `lobpcg` is also returned by `solve_discretised` itself. 
In your implementation replace `tol32=XXX` by a sensible default value for `tol32`.

Explain why you have to recompute the Hamiltonian with `T=Float64` instead of simply converting the `Float32` Hamiltonian. Is there a way to avoid computing the Hamiltonian twice ?
"""

# ╔═╡ 5b7b3343-1104-4856-b250-0435a3c770f9
function solve_discretised(V, N; n_ep=3, tol32=1e-1, tol=1e-6, maxiter=100)
	H64 = fd_hamiltonian_periodic(V, N; T=Float64)
	H32 = Float32.(H64)

	Pinv32 = InverseMap(factorize(H32))
	intermediate_result = lobpcg(H32, Pinv=Pinv32, X=randn(eltype(H32), size(H32, 2), n_ep), tol=tol32, maxiter=maxiter, verbose=false)
	iter32 = size(intermediate_result.λ, 1)
	
	Pinv64 = InverseMap(factorize(H64))
	final_result = lobpcg(H64, Pinv=Pinv64, X=intermediate_result.X, tol=tol, maxiter=maxiter-iter32, verbose=false)
	return final_result
end

# ╔═╡ 0575ec37-c8ff-46f1-b8a7-ad1d83b511ca
let
	plot(last.(solve_discretised(cos, 4000).residual_norms), yscale=:log10)
	xlabel!("iteration number")
	ylabel!("residual norm")
	title!("residual norm evolution with mixed precision routine")
end

# ╔═╡ 1f5203a7-697e-40de-b5ba-0d44013a0a51
md"-----------------"

# ╔═╡ 776df9aa-f137-41a9-a721-af70fe6439bc
md"""
### Task 1.4: Getting the answer as fast as possible

**(a)** Combine the ideas of Task 1.1 (e), Task 1.2 (c) and Task 1.3 (b), that is the use of 
- perturbation theory
- preconditioning
- mixed precision
to design a diagonalisation algorithm, which finds the first three eigenvalues of a discretised version of $\textbf{H}_\text{per}$ as fast as possible. Test your algorithm for various values of $N$. Explain how you achieved this algorithm and what were your thoughts in its design.

Indicate clearly in your notebook the final version of your algorithm and test it for the case of $N = 10000$, achieving a tolerance of `tol=1e-8` for the first 3 eigenpairs. Benchmark it using `@btime` and indicate the total time it took for you to achieve this task (including the time to setup intial guesses, preconditioners etc.). How much time do you need to converge the first three eigenpairs to $6$ digits?
"""

# ╔═╡ 87bff49e-d791-482a-94c9-c1486a4ae094
# Solve H_dir as fast as possible:
function solve_H_dir_fast(V, N; n_ep=3, tol32=1e-1, tol=1e-6, maxiter=100)
	H64=fd_hamiltonian_dirichlet(V, N, T=Float64)
	H32=Float32.(H64)
	Pinv=InverseMap(factorize(H32))

	#the lobpcg_new allows the use of a less precise preconditioner, downgrading the computation of the residual to Float32, as this precision is sufficient in our case. This makes lobpcg much faster without compromising the precision of the result.
	intermediate_result = lobpcg_new(H32, Pinv=Pinv, X=randn(eltype(H32), size(H32, 2), n_ep), tol=tol32, maxiter=maxiter, verbose=false)
	iter32 = size(intermediate_result.λ, 1)
	
	final_result = lobpcg_new(H64, Pinv=Pinv, X=intermediate_result.X, tol=tol, maxiter=maxiter-iter32, verbose=false)
	return final_result
	
end

# ╔═╡ 4c33a6ca-5be0-44e7-939d-b07c96e0999f
let
	result = @btime solve_H_dir_fast(cos, 10000, tol=1e-8)
	plot(last.(result.residual_norms), yscale=:log10)
end

# ╔═╡ 3bf2daa3-5920-408a-b32e-0b5c25cd9580
md"-----------------"

# ╔═╡ 0ecd527b-da61-4737-9f05-5f4c851a6077
md"""
## Part 2: Generalisation and error estimates

Now we will generalise the problem we considered in Part 1 to three dimensions and general periodic Hamiltonians. Our setup is again the class of Hamiltonians
```math
H = -\frac12 \Delta + V,
```
where $V$ is a potential, which is periodic
with respect to some lattice $\mathbb{L}$.
That is that $V(x + R) = V(x)$ for all vectors $R \in \mathbb{L}$ with
```math
\mathbb{L} = \left\{ g_1 a_1 + g_2 a_2 + g_3 a_3 \ \middle| \ g_1, g_2, g_3 \in \mathbb{Z} \right\}
```
and $a_1$, $a_2$, $a_3$ are given vectors from $\mathbb{R}^3$.
For this setting we will further develop mathematically guaranteed
estimates for the total numerical error of our eigenvalue computations,
such that the exact answer is provably within our error bound.

For our numerical computation in this project we will mostly employ the density-functional toolkit (DFTK), which has already been installed alongside this notebook. You can find the source code on [dftk.org](https://dftk.org) and documentation as well as plenty of usage examples on [docs.dftk.org](https://docs.dftk.org).
"""

# ╔═╡ 6bdc0066-1f86-458d-aee5-6a7564f339e6
md"""
Here we will provide a very concise summary of the key ingredients for modelling periodic problems. See <https://docs.dftk.org/stable/guide/periodic_problems/>
for a gentle introduction to periodic potentials in one dimension.
We will also pick up on these points in more details in the [lecture about periodic problems](https://teaching.matmat.org/error-control/11_Periodic_problems.html), but you should already make yourself a little familiar with this topic by reading the referenced resources, such that you can start working on this project.

It turns out that in order to characterise the spectrum of $H$
a computationally and physically very advantageous approach is
to employ Bloch-Floquet theory (the Bloch transformation).
Essentially the outcome is that in order to obtain the spectrum of $H$
one can alternatively introduce so-called Bloch fibers
```math
H_k = \frac{1}{2}(-i \nabla_x + k)^2
```
where the $k$-points are taken from a specific set $\overline{\Omega^\ast}$ discussed in more detail below.
Crucially the union of the spectra of all Bloch fibers reproduces the spectrum of $H$,
i.e. more mathematically
```math
\sigma(H) = \bigcup_{k \in \overline{\Omega^\ast}} \sigma(H_k).
```

Of main interest and the focus of our work is the so-called **band structure**,
that is the dependency of the few lowest eigenvalues of the $H_k$ on $k$.
More precisely having found some eigenstates
```math
\begin{align}
H_k \phi_{k,n} &= λ_{k,n} \phi_{k,n} && k \in \overline{\Omega^*}, n\in \mathbb N
\\
\langle \phi_{k,n} , \phi_{k,m} \rangle &= \delta_{mn} && \forall n,m \in \mathbb N.
\end{align}
```
with the usual sorting for each $k \in \overline{\Omega^*}$ : $λ_{k,1} ≤ λ_{k,2} ≤ λ_{k,3} ≤ …$, we define the mapping  for each $n \in \mathbb N$:
```math
λ_n : \overline{\Omega^*} \to \mathbb R : k ↦ λ_{k,n}.
```
The latter object is called the $n$-th **energy band**.

At the end of this project you will have coded up a routine to
compute the band structure of simple periodic potentials $V$ in a way
that the error is fully tracked and mathematically guaranteed
error bars can be annotated to the computed band structure.
"""

# ╔═╡ d3e0a2e9-9d89-476b-8d37-da7b14904b1e
md"""
----

As example systems in this work we will only consider face-centred cubic crystals in the diamond structure. As the name suggests this family of crystals includes diamond, but also materials such as bulk silicon, germanium or tin. For these the lattice
```math
\mathbb{L} = \mathbb{Z} a_1 + \mathbb{Z} a_2 + \mathbb{Z} a_3
```
can be constructed from the unit cell vectors
```math
a_1 = \frac{a}{2} \left(\begin{array}{c} 0\\1\\1 \end{array}\right)\qquad
a_2 = \frac{a}{2} \left(\begin{array}{c} 1\\0\\1 \end{array}\right)\qquad
a_3 = \frac{a}{2}\left(\begin{array}{c} 1\\1\\0 \end{array}\right)
```
where $a$ is the lattice constant, which differs between the materials. Keeping the notation from the lecture we will use $\Omega$ to refer to the unit cell
```math
\begin{align*}
\Omega &= \left\{x \in \mathbb{R}^3 \,\middle|\, |x - R| > |x| \quad \forall R \in \mathbb{L}\setminus\{0\}\right\} \\
&= \left\{ x_1 a_1 + x_2 a_2 + x_3 a_3 \ \middle|\ -\frac12 < x_1 < \frac12 \text{ and } -\frac12 < x_2 < \frac12 \text{ and } -\frac12 < x_3 < \frac12 \right\},
\end{align*}
```
and employ $\mathbb{L}^\ast$ to refer to the reciprocal lattice with vectors $b_1$, $b_2$, $b_3$ with $a_i \cdot b_j = 2π δ_{ij}$. $\Omega^\ast$ denotes the first Brillouin zone (unit cell of $\mathbb{L}^\ast$), that is
```math
\Omega^\ast = \left\{ k_1 b_1 + k_2 b_2 + k_3 b_3 \ \middle|\ 
-\frac12 < k_1 < \frac12 \text{ and } -\frac12 < k_2 < \frac12 \text{ and } -\frac12 < k_3 < \frac12
 \right\}.
```
We will also need the closure of this set, where the boundary is also included:
```math
\overline{\Omega^\ast} = \left\{ k_1 b_1 + k_2 b_2 + k_3 b_3 \ \middle|\ 
-\frac12 ≤ k_1 ≤ \frac12 \text{ and } -\frac12 ≤ k_2 ≤ \frac12 \text{ and } -\frac12 ≤ k_3 ≤ \frac12
 \right\}.
```
"""

# ╔═╡ d12f9337-0b87-40cb-817d-8a4ac3858196
md"""
For the discretisation (see also <https://docs.dftk.org/stable/guide/discretisation/>) we introduce the normalised plane waves
```math
e_G(x) = \frac{e^{iG \cdot x}}{\sqrt{|\Omega|}}\quad \forall G \in \mathbb{L}^\ast,
```
which allow to express the Fourier series of the potential as
```math
V(x) = \frac{1}{\sqrt{|\Omega|}} \sum_{G \in \mathbb{L}^\ast} \hat{V}(G) e^{i G \cdot x}
= \sum_{G \in \mathbb{L}^\ast} \hat{V}(G) e_G(x).
```
To simplify our notation we will assume that $G$-vectors always run over the countably infinite reciprocal lattice $\mathbb{L}^\ast$ without making explicit reference to this set.

An early model to describe these materials theoretically is the Cohen-Bergstresser model[^CB1966]. The special and simplifying property of this model is that only a relatively small number of Fourier coefficients is non-zero. In particular we can define a cutoff $\mathcal{E}_V$ such that
```math
V(x) = \sum_{|G| < \sqrt{2\mathcal{E}_V}} \hat{V}(G) e_G(x),
```
where each $\hat{V}(G)$ is finite.

[^CB1966]: M. L. Cohen and T. K. Bergstresser Phys. Rev. **141**, 789 (1966) DOI [10.1103/PhysRev.141.789](https://doi.org/10.1103/PhysRev.141.789)

"""

# ╔═╡ 354b8072-dcf0-4182-9897-c3e9534bef5a
md"""
### Task 2.1: Cohen-Bergstresser Hamiltonians

**(a)** Using the results of the [lecture on periodic problems](https://teaching.matmat.org/error-control/11_Periodic_problems.html) show that $H$ is self-adjoint on $L^2(\mathbb{R}^3)$ with the domain $D(H) = H^2(\mathbb{R}^3)$ if $V$ is the Cohen-Bergstresser potential.   
*Hint:* You should attempt this exercise after lecture 11. Show and use that the Cohen-Bergstresser potential function is bounded and $\mathbb{L}$-periodic.
"""

# ╔═╡ 9c98cecd-53b9-483e-aecf-c925b5bfd18a
md"""
We first show that the Cohen-Bergstresser potential function is bounded and $\mathbb{L}$-periodic. Recall
```math
V(x) = \sum_{|G| < \sqrt{2\mathcal{E}_V}} \hat{V}(G) e_G(x),
```
The $\mathbb{L}$-periodicity of $V$ follows from the $\mathbb{L}$-periodicity of the plane wave basis functions $e_G$. For boundedness, observe
```math
\begin{align*}
\sup_{x \in \mathbb{R}^3} | V(x) | &\leq \sum_{|G| < \sqrt{2\mathcal{E}_V}} |\hat{V}(G) | \cdot 
 || e_G(x) || \\
&\leq \sum_{|G| < \sqrt{2\mathcal{E}_V}} |\hat{V}(G)| < \infty
\end{align*}
```
Therefore, $V(x)$ is a bounded $\mathbb{L}$-periodic function. By theorem 1 on the lecture on periodic problems, it follows that $H = -\frac{1}{2} \nabla + V$ is self-adjoint with domain $D(H) = H^2(\mathbb{R}^3)$.
"""

# ╔═╡ ef2796f3-b6d4-4403-aaa6-ed3d9d541a3a
md"""
------
A consequence of *(a)* is that the Bloch-Floquet transformation yields Bloch fibers
```math
H_k = T_k + V \quad \text{where} \quad T_k = \frac{1}{2}(-i \nabla_x + k)^2
```
which are self-adjoint on $L^2_\text{per}(\Omega)$ with $D(H_k) = H^2_\text{per}(\Omega)$. The natural inner product for our problem is thus the $\langle \, \cdot\, | \, \cdot \, \rangle_{L^2_\text{per}(\Omega)}$ inner product, which is the one we will employ unless a different one is specified.

For a particular $k$-point we can discretise the problem using the plane-wave basis
```math
\mathbb{B}_k^{\mathcal{E}} = \left\{ e_G \, \middle| \, G \in \mathbb{L}^\ast \ \text{ and } \frac12 |G+k|^2 < \mathcal{E} \right\}
```
where $\mathcal{E}$ is a chosen cutoff.
"""

# ╔═╡ 7495739f-56b0-4076-8c91-f4155d2ba00f
md"""
**(b)** Show that 
```math
\langle e_G | e_{G'} \rangle = \delta_{GG'}
```
and that
```math
\langle e_G | T_k e_{G'} \rangle = \delta_{GG'} \frac12 |G+k|^2.
```
From this conclude that for any $e_G \in \mathbb{B}_k^{\mathcal{E}}$ we have
```math
|\Delta G| > \sqrt{2\mathcal{E}_V} \quad \Longrightarrow \quad
\langle e_{G + \Delta G} | H e_{G} \rangle = 0, \qquad (1)
```
i.e. that each plane wave only couples via the Hamiltonian with these plane waves, that differ in the wave vector by no more than $\sqrt{2\mathcal{E}_V}$.
"""

# ╔═╡ fd4691d1-b46c-444a-96f6-443369634d59
md"""
First equality:
```math
\begin{align*}
	\langle e_G | e_{G'} \rangle &= \frac{1}{|\Omega|} \int_\Omega e^{i G \cdot x} e^{- i G' \cdot x} dx \\
	&= \frac{1}{|\Omega|} \int_\Omega e^{i (G - G')' \cdot x} dx \\
	&= \delta_{G G'}
\end{align*}
```
since $e^{i G \cdot x}|_{\delta \Omega}$ is constant.
Second equality:
```math
\begin{align*}
	\langle e_G | T_k e_{G'} \rangle &= \frac{1}{2|\Omega|} \int_\Omega e^{i G \cdot x} \overline{(-i \Delta_x + k)^2 e^{i G' \cdot x}} dx \\
	&= \frac{1}{2|\Omega|} \int_\Omega e^{i G \cdot x} \overline{(-i \Delta_x + k) (G' +k) e^{i G' \cdot x}} dx \\
	&= \frac{1}{2|\Omega|} \int_\Omega e^{i G \cdot x} \overline{(G' + k) \cdot (G' +k) e^{i G' \cdot x}} dx \\
	&= \frac{1}{2|\Omega|} \int_\Omega e^{i G \cdot x} |G' + k|^2 e^{-i G'\cdot x} dx \\
	&= \delta_{GG'} \frac12 |G+k|^2.
\end{align*}
```
where the second and third equalities use the relation $(-i \Delta_x + k) e^{i G' \cdot x} = (G' + k) e^{i G' x}$ and the second $\Delta_x$ is interpreted as a divergence operator. Conclusion: For any $e_G \in \mathbb{B}_k^{\mathcal{E}}$ and $|\Delta G| > \sqrt{2\mathcal{E}_V}$, 
```math
\begin{align*}
	\langle e_{G + \Delta G} | H_k e_{G} \rangle &= \langle e_{G + \Delta G} | V e_{G} \rangle \tag{by above} \\
&= \sum_{|G'| < \sqrt{2\mathcal{E}_V}} \overline{\hat{V}(G')} \langle e_{G + \Delta G}, e_{G + G'} \rangle \\
&= \sum_{|G'| < \sqrt{2\mathcal{E}_V}} \overline{\hat{V}(G')} \langle e_{\Delta G}, e_{G'} \rangle \\
&= 0
\end{align*}
```
since $\langle e_{\Delta G}, e_{G'} \rangle = 0$ for all $|G'| < \sqrt{2\mathcal{E}_V}$.
"""

# ╔═╡ 3f128396-f3b6-44b0-906e-b1a8467fd43f
md"""
-------
"""

# ╔═╡ 05d40b5e-fd83-4e73-8c78-dfd986a42fc0
md"""
## Familiarisation and Bauer-Fike estimates

Work through the documentation pages
- <https://docs.dftk.org/stable/guide/periodic_problems/>,
- <https://docs.dftk.org/stable/guide/discretisation/> as well as
- <https://docs.dftk.org/stable/guide/tutorial/>
to get some basic familiarity with DFTK. 

Following loosely [another documented example](https://docs.dftk.org/stable/examples/cohen_bergstresser/), this code sets up a calculation of silicon using the Cohen-Bergstresser potential and $\mathcal{E} = 10$.
"""

# ╔═╡ c4393902-7c57-4126-80af-8765bea42ebd
begin
	Si = ElementCohenBergstresser(:Si)
	atoms = [Si, Si]
	positions = [ones(3)/8, -ones(3)/8]
	lattice = Si.lattice_constant / 2 .* [[0 1 1.]; [1 0 1.]; [1 1 0.]]
end;

# ╔═╡ 78cc8d4a-cb63-48d0-a2f9-b8ec8c2950e5
begin
	model = Model(lattice, atoms, positions; terms=[Kinetic(), AtomicLocal()])
	basis_one = PlaneWaveBasis(model; Ecut=10.0, kgrid=(1, 1, 1));
end

# ╔═╡ 0aaeb747-ce4f-4d53-a3b7-f8019c3d688b
md"""
The `kgrid` parameter selects the $k$-point mesh $\mathbb{K} \subset \overline{\Omega^\ast}$, which is employed by determining the number of $k$-points in each space dimension. In this case only a single $k$-point is used, namely the origin of $\overline{\Omega^\ast}$:
"""

# ╔═╡ ffd75a52-9741-4c12-ae2f-303db8d70332
basis_one.kpoints

# ╔═╡ 0f6a001d-ce82-4d4b-a914-37bf0b62d604
md"""
The origin of the Brillouin zone has a special name and is usually called $\Gamma$-point. Before treating the case of band structure computations (where $k$ is varied), we stick to using only a single $k$-point.

Each `KPoint` in DFTK automatically stores a representation of the basis $\mathbb{B}_k^{\mathcal{E}}$. If you are curious, the $G$-vectors of the respective plane waves $e_G$ can be looked at using the functions
"""

# ╔═╡ 66b93963-272f-4e1b-827c-0f3b551ee52b
G_vectors_cart(basis_one, basis_one.kpoints[1])  # G-vectors of 1st k-point

# ╔═╡ dfe6be2f-fb6b-458b-8016-8d89dfa66ed9
md"""
Similarly a discretised representation of $H$ restricted to the fibres matching the chosen $k$-point mesh can be obtained using
"""

# ╔═╡ 252e505d-a43d-472a-9fff-d529c3a2eeb7
ham_one = Hamiltonian(basis_one)

# ╔═╡ 0f5d7c6c-28d7-4cf9-a279-8117ddf7d9d5
md"""
Let us denote by $P_k^\mathcal{E}$ the projection
into the plane-wave basis $\mathbb{B}_k^\mathcal{E}$
and further by
$H_k^{\mathcal{E}\mathcal{E}} \equiv P_k^\mathcal{E} H_k P_k^\mathcal{E}$
the discretised fibers with elements
```math
\left(H_k^{\mathcal{E}\mathcal{E}}\right)_{GG'}
= \left(P_k^\mathcal{E} H_k P_k^\mathcal{E}\right)_{GG'}
= \langle e_G,\, H_k\, e_{G'} \rangle
\qquad \text{for $G, G' \in \mathbb{B}_k^\mathcal{E}$}.
```
Representations of $H_k^{\mathcal{E}\mathcal{E}}$ in DFTK
are available via indexing of `Hamiltonian` objects.
E.g. `ham[1]` corresponds to the fiber of `basis.kpoints[1]`,
`ham[2]` to `basis.kpoints[2]` and so on.
This can also be seen from comparing their sizes with the
size of $\mathbb{B}_k^\mathcal{E}$ (available via the `G_vectors_cart`):
"""

# ╔═╡ b4ffe212-6f71-4d61-989a-570c5448877a
begin
	@show length(G_vectors_cart(basis_one, basis_one.kpoints[1]))
	@show size(ham_one[1])
	@show typeof(ham_one[1])
end

# ╔═╡ 5138009b-89e6-4863-b8ec-6ef0e86e2124
md"""
These `DftHamiltonianBlock` objects very much behave like matrices,
e.g. they can be multiplied with vectors, but also used in iterative solvers.

The `diagonalize_all_kblocks` function does some optimisations and book-keeping to solve for the `n_bands` lowest eigenpairs at each $H_k$. Here we define a wrapper to simplify operations:
"""

# ╔═╡ 25775046-d3bc-4e65-80c2-3fa8021b5d86
function diagonalize_auto(ham, n_bands; solver=DFTK.lobpcg_hyper)
	# diagonalize_all_kblocks returns eigenvalues under the key λ
	# and eigenvectors under the key X
	(; λ, X) = diagonalize_all_kblocks(solver, ham, n_bands)
	# Rename the two keys for consistency with the compute_bands function,
	# which we will use later.
	(; eigenvalues=λ, ψ=X)
end

# ╔═╡ 0f3ed8c8-0978-4b7e-a9bd-de902783201a
md"""
We use this function to solve for $6$ eigenpairs:
"""

# ╔═╡ c3a6aec7-ab22-4017-8e2f-bfe5696021dd
begin
	n_bands = 6
	eigres_one = diagonalize_auto(ham_one, n_bands)
end

# ╔═╡ 8f3634ca-9c13-4d40-af3c-0c80588fd8c8
md"""
The returned object `eigres_one` now contains the eigenvalues in `eigres_one.eigenvalues` and the Bloch waves in `eigres_one.ψ`. Again both are a vector along the $k$-points, further `eigres_one.ψ[ik]` is of size `length(G_vectors_cart(basis, kpoint))` $\times$ `n_bands`:
"""

# ╔═╡ ac0cffd2-effb-489c-a118-60864798d55e
begin
	@show size(eigres_one.ψ[1])
	@show (length(G_vectors_cart(basis_one, basis_one.kpoints[1])), n_bands)
end

# ╔═╡ 142ac96e-bd9d-45a7-ad31-e187f28e884a
md"""
As a result if we wanted to compute at each $k$-point the residual
```math
H_k^{\mathcal{E}\mathcal{E}} \widetilde{X}_{kn} - \widetilde{λ}_{kn} \widetilde{X}_{kn}
\qquad (2)
```
after we have been able to diagonalise $H_k^{\mathcal{E}\mathcal{E}}$,
we can simply iterate as such:
"""

# ╔═╡ 601c837c-1615-4826-938c-b39bb35f46d1
for (ik, kpt) in enumerate(basis_one.kpoints)
	hamk = ham_one[ik]
	λk   = eigres_one.eigenvalues[ik]
	Xk   = eigres_one.ψ[ik]

	residual_k = hamk * Xk - Xk * Diagonal(λk)
	println(ik, "  ", norm(residual_k))
end

# ╔═╡ e0a07aca-f81a-436b-b11e-8446120e0235
md"""
### Task 2.2: $\Gamma$-point calculations

**(a)** For the case of employing only a single $k$-point (`kgrid = (1, 1, 1)`) vary $\mathcal{E}$ (i.e. `Ecut`) between $5$ and $30$. Taking $\mathcal{E} = 80$ as a reference, plot the convergence of the first two eigenpairs of $H_k$ at the $\Gamma$ point.
"""

# ╔═╡ 56108a52-61ed-43fc-bd0d-7fc4221f05ce
function calculate_eigenvalues(model, Ecut)
    basis = PlaneWaveBasis(model; Ecut=Ecut, kgrid=(1, 1, 1))
    ham = Hamiltonian(basis)
    eigres = diagonalize_auto(ham, 2)
    return eigres.eigenvalues
end

# ╔═╡ 580e4dc0-b4dd-4f63-9db7-7fdebf225f1c
begin
	# Define a range of Ecut values to test
	Ecut_values = 5:0.5:30
end

# ╔═╡ bfcf40d8-3855-454b-843b-fe3bdd1e5a92
begin
	# Initialize a matrix to store the results
	eigenvalues = zeros(length(Ecut_values), 2)
end

# ╔═╡ 20c15f08-8bf9-4c6b-84ab-4486ba0a9237
# Loop over Ecut values and calculate eigenvalues
for (i, Ecut) in enumerate(Ecut_values)
    eigenvalues[i, :] = calculate_eigenvalues(model, Ecut)[1]
end

# ╔═╡ e0c6e8d3-4f0a-4be6-805f-79fb012c01d8
begin
	# Plot the results
	plot(Ecut_values, eigenvalues[:, 1], label="First Eigenvalue")
	#plot!(Ecut_values, eigenvalues[:, 2], label="Second Eigenvalue")
	xlabel!("Ecut")
	ylabel!("Eigenvalue")
	title!("Eigenvalues vs. Ecut")
end

# ╔═╡ 0e0ed0c6-ab11-4d4e-9552-3c89f7bd1392
begin
	# Plot the results
	#plot(Ecut_values, eigenvalues[:, 1], label="First Eigenvalue")
	plot(Ecut_values, eigenvalues[:, 2], label="Second Eigenvalue")
	xlabel!("Ecut")
	ylabel!("Eigenvalue")
	title!("Eigenvalues vs. Ecut")
end

# ╔═╡ 5daa306b-0dfe-4d4b-9e03-1ed7b50d6c61
calculate_eigenvalues(model, 80)[1]

# ╔═╡ ba1ba08f-7fe5-4fb3-b736-adaf7200af08
md"""-------------"""

# ╔═╡ 58856ccd-3a1c-4a5f-9bd2-159be331f07c
md"""
We want to compare this convergence behaviour with a first estimate of the eigenvalue discretisation error based on the Bauer-Fike bound. Given an approximate eigenpair $(\widetilde{λ}_{kn}, \widetilde{X}_{kn})$ of the fiber $H_k$ we thus need access to the residual
```math
r_{kn} = H_k \widetilde{X}_{kn} - \widetilde{λ}_{kn} \widetilde{X}_{kn}.
```
If we performed our computation
using the plane-wave basis $\mathbb{B}_k^\mathcal{E}$,
then by construction
```math
P_k^\mathcal{E} \widetilde{X}_{kn} = \widetilde{X}_{kn},
```
such that
```math
\begin{align*}
P_k^\mathcal{E} r_{kn} &= P_k^\mathcal{E} H_k P_k^\mathcal{E}\, P_k^\mathcal{E} \widetilde{X}_{kn} - \widetilde{λ}_{kn} P_k^\mathcal{E} \widetilde{X}_{kn} \\
&= H_k^{\mathcal{E}\mathcal{E}} \widetilde{X}_{kn} - \widetilde{λ}_{kn} \widetilde{X}_{kn},
\end{align*}
```
which is exactly the quantity we computed in $(2)$ above.
In other words $P_k^\mathcal{E} r_{kn}$ is the residual corresponding to the
iterative diagonalisation, thus leading to the **algorithm error**.

However, we are also interested in the **discretisation error**, which we obtain from the missing residual term in $(2)$, namely 
```math
Q_k^\mathcal{E} r_{kn} = (1 - P_k^\mathcal{E}) r_{kn} = Q_k^\mathcal{E} H_k P_k^\mathcal{E} \widetilde{X}_{kn} = H_k^{\mathcal{E}^\perp\mathcal{E}} \widetilde{X}_{kn}
```
where $Q_k^\mathcal{E} = (1-P_k^\mathcal{E})$ is the orthogonal projector to $P_k^\mathcal{E}$ and we introduced the notation
```math
H_k^{\mathcal{E}^\perp\mathcal{E}} = Q_k^\mathcal{E} H_k P_k^\mathcal{E}.
```


While computing $P_k^\mathcal{E} r_{kn}$ is generally easy (we just did it above in 5 lines), obtaining $Q_k^\mathcal{E} r_{kn}$ is impossible for general potentials $V$: since $H_k P_k^\mathcal{E} \widetilde{X}_{kn}$ can have support anywhere on $L^2_\text{per}(\Omega)$, estimating $Q_k^\mathcal{E} r_{kn}$ usually requires making a mathematical statement about the behaviour of $H_k$ on all of $L^2_\text{per}(\Omega)$. Estimating discretisation errors is thus substantially more challenging than estimating algorithm errors.
"""

# ╔═╡ e04087db-9973-4fad-a964-20d109fff335
md"""
**(b)** In our case of Cohen-Bergstresser Hamiltonians we are more lucky. Using $(1)$ show that there exists a cutoff $\mathcal{F} > \mathcal{E}$, such that
```math
r_{kn} = P_k^\mathcal{F} r_{kn} = H_k^{\mathcal{F}\mathcal{F}} \widetilde{X}_{kn} - \widetilde{λ}_{kn} \widetilde{X}_{kn}.
```
Having diagonalised the Hamiltonian using cutoff $\mathcal{E}$ we can thus obtain the *full residual* by a computation of the Hamiltonian-vector product using an elevated cutoff $\mathcal{F}$.
"""

# ╔═╡ 83b6b542-0eb1-4ff7-bea6-26466af478f5
md"""
It suffices to show
```math
H_k \widetilde{X}_{kn} = H_k^{\mathcal{F} \mathcal{F}} \widetilde{X}_{kn}
```
for some $\mathcal{F}$ large enough. Recall $\widetilde{X}_{kn}$ was computed in the plane-wave basis $\mathbb{B}_k^\mathcal{E}$. By (1), we know that if $e_G \in \mathbb{B}_k^\mathcal{E}$ and $|\Delta G| > \sqrt{2 \mathcal{E}_V}$ then $\langle e_{G + \Delta G} | H_k e_{G} \rangle = 0$, so
```math
\langle e_{G + \Delta G} | H_k \widetilde{X}_{kn} \rangle = 0
```
so we can take $\mathcal{F} = \mathcal{E} + \mathcal{E}_V$, because then if $e_G \not\in \mathbb{B}_k^\mathcal{F}$, then $\frac{1}{2} |G + k|^2 \geq \mathcal{E} + \mathcal{E}_V$ and so $G$ can be written as $G = G' + \Delta G$ for some $e_{G'} \in \mathbb{B}_k^\mathcal{E}$ and $|\Delta G| > \sqrt{2 \mathcal{E} V}$. Thus, $H_k \widetilde{X}_{kn}$ can be expressed entirely in the basis $\mathbb{B}_k^\mathcal{F}$ and so
```math
H_k \widetilde{X}_{kn} = H_k^{\mathcal{F} \mathcal{F}} \widetilde{X}_{kn}
```
as required.
"""

# ╔═╡ abdfcc43-245d-425a-809e-d668f03e9b45
md"""
**(c)** In DFTK nobody stops you from using multiple bases of different size. Moreover, having obtained a set of bloch waves `X_small` using `basis_small` you can obtain its representation on the bigger `basis_large` using the function
```julia
X_large = transfer_blochwave(X_small, basis_small, basis_large)
```
Using this technique you can compute the application of the Hamiltonian using a bigger basis (just as `Hamltonian(basis_large) * X_large`). Use this setup to vary $\mathcal{F}$ and use this to estimate $\mathcal{E}_V$ numerically. Check your estimate for various values of $\mathcal{E}$ to ensure it is consistent. Rationalise your results by taking a look at the [Cohen Bergstresser implementation in DFTK](https://github.com/JuliaMolSim/DFTK.jl/blob/0b61e06db832ce94f6b66c0ffb1d215dfa4822d4/src/elements.jl#L142).
"""

# ╔═╡ ceb72ee7-f7ac-4a49-a585-78e1d55cde2a
# Your answer here

# ╔═╡ d26fec73-4416-4a20-bdf3-3a4c8ea533d1
md"""
**(d)** Based on the Bauer-Fike bound estimate the algorithm and arithmetic error for the first two eigenpairs at the $\Gamma$ point and for using cutoffs between $\mathcal{E} = 5$ and $\mathcal{E} = 30$. Add these estimates to your plot in $(a)$. What do you observe regarding the tightness of the bound ?
"""

# ╔═╡ b1270539-8079-4761-ab61-55d08d563635
# Your answer here

# ╔═╡ 9a953b4e-2eab-4cb6-8b12-afe754640e22
md"""----"""

# ╔═╡ 0616cc6b-c5f8-4d83-a247-849a3d8c5de8
md"""
## Band structure computations

Next we turn our attention to band structure computations. The $n$-th **band** is the mapping from $k$ to the $n$-th eigenvalue $λ_{kn}$ of the Bloch fiber $H_k$. For standard lattices --- like the diamond-FCC lattices we consider --- there are tabulated standard 1D paths through the Brillouin zone $\Omega^\ast$, that should be used to maximise the physical insight one may gain from the computation.

In DFTK this path can be automatically determined. The following function extracts such a path (the warnings can be ignored in this function call):
"""

# ╔═╡ d363fa0a-d48b-4904-9337-098bcef015bb
kpath = irrfbz_path(model)

# ╔═╡ 9e7d9b0f-f29d-468f-8f7c-e01f271d57c6
md"""
Based on this data we can use the `compute_bands` function to diagonalise all $H_k$ along the path using a defined cutoff `Ecut`. 
"""

# ╔═╡ d34fa8e5-903f-45f4-97f8-32abd8883e9f
function compute_bands_auto(model; Ecut, kwargs...)
	basis = PlaneWaveBasis(model; Ecut, kgrid=(1, 1, 1))
	compute_bands(basis, kpath; kline_density=15, kwargs...)
end

# ╔═╡ e3244165-9f04-4676-ae80-57667a6cb8d5
band_data = compute_bands_auto(model; n_bands=6, tol=1e-3, Ecut=10)

# ╔═╡ 968499a6-c0c3-484b-a006-9bbdccca62ba
md"""
The resulting object contains many details about the band computation.
In particular `compute_bands` internally also calls `diagonalize_all_kblocks`, such that a similar information is available.

For example we obtain the eigenvalues of the 3rd $k$-point
"""

# ╔═╡ 61448d06-c628-493b-8fa0-a8f0a41acd1d
band_data.basis.kpoints[3]

# ╔═╡ 722f34dd-e1cb-4a3c-9a13-7ff6117029fc
md"using the expression"

# ╔═╡ 06270fe2-df78-431a-8433-d69fc89ae5c3
band_data.eigenvalues[3]

# ╔═╡ 01c80f21-b18b-4f84-9758-ada4895095f1
md"""and the corresponding eigenvectors as """

# ╔═╡ 586e5ae6-f0f8-4222-8a7c-aa687c886685
band_data.ψ[3]

# ╔═╡ e56b2164-5f66-4745-b1a5-711dcc1a324b
md"Once the data has been computed, a nice bandstructure plot can be produced using"

# ╔═╡ 66992c0d-262c-4211-b5d2-1ba247f9e8ba
plot_bandstructure(band_data)

# ╔═╡ 86f70a52-13ce-42b5-9a28-2c01d92b022d
md"Error bars for indicating eigenvalue errors can also be easily added:"

# ╔═╡ 2dee9a6b-c5dd-4511-a9e3-1d5bc33db946
begin
	eigenvalues_error = [0.02 * abs.(randn(size(λk)))
		                 for λk in band_data.eigenvalues]  # dummy data
	data_with_errors = merge(band_data, (; eigenvalues_error))
	plot_bandstructure(data_with_errors)
end

# ╔═╡ 6986bb41-aed6-4ca6-aa70-90cb1fe0bfc3
md"""Note, that DFTK here explicitly checks for the key `eigenvalues_error`, so make sure your error values are supplied under that key in the named tuple. E.g. use the pattern
```julia
data_with_errors = merge(band_data, (; eigenvalues_error=my_crazy_errors))
```
if the error values are stored in the variable `my_crazy_errors`.
"""

# ╔═╡ da8c23eb-d473-4c2f-86f9-879016659f3e
md"""
### Task 2.3: Bands with Bauer-Fike estimates

For $\mathcal{E} = 7$ and the $6$ lowest eigenpairs plot a band structure in which you annotate the bands with error bars estimated from Bauer-Fike. To compute the Hamiltonian-vector product employing a larger basis set with cutoff `Ecut_large`, employ the following piece of code, which forwards the $k$-Point coordinates from one basis to another:
"""

# ╔═╡ 3006a991-d017-40b3-b52b-31438c05d088
function basis_change_Ecut(basis_small, Ecut_large)
	PlaneWaveBasis(basis_small.model; Ecut=Ecut_large, basis_small.kgrid)
end

# ╔═╡ 646242a2-8df6-4caf-81dc-933345490e6c
md"""
-------
"""

# ╔═╡ 1957a8bb-b426-49c3-be25-2ad418d292ad
# Your answer here

# ╔═╡ d7f3cadf-6161-41ce-8a7a-20fba5182cfb
md"""
## Kato-Temple estimates

Following up on Task 2.2 (d) we want to improve the bounds using Kato-Temple estimates. Similar to our previous developments in this regard the challenging ingredient is estimating a lower bound to the gap. If $λ_{kn}$ denotes the eigenvalue closest to the approximate eigenvalue $\widetilde{λ}_{kn}$, we thus need a lower bound to
```math
δ_{kn} = \min_{s \in \sigma(H_k) \backslash \{λ_{kn}\}} |s - \widetilde{λ}_{kn}|.
```

The usual approach we employed so far (see e.g. Sheet 6) was to assume that our obtained approximations $\widetilde{λ}_{kn}$ for $n = 1,\ldots,N$ did not miss any of the exact eigenvalues. With this assumption we would employ a combination of Kato-Temple and Bauer-Fike bound to obtain a lower bound to $δ_{kn}$.

Already for estimating the eigenvalue error of iterative eigensolvers, this can be a far-fetched assumptions as we saw in previous exercises. For estimating the discretisation error where the discretisation by nature of projecting into a finite-dimensional space inevitably changes the properties of the spectrum, this is even more far fetched. In fact in practice it is not rare that an unsuited basis changes the order of eigenpairs, completely neglects certain eigenpairs etc. Since the goal of *a posteriori* estimation is exactly to detect such cases, we will in Task 2.5 also develop techniques based on weaker assumptions.

Before doing so, we first extend the usual technique to infinite dimensions.
"""

# ╔═╡ 5f0d20d1-c5f0-43b8-b65d-d5a2039cd01a
md"""
### Task 2.4: Simple and non-guaranteed Kato-Temple estimates

**(a)** Notice that the Kato-Temple theorem requires the targeted eigenvalue $λ_{kn}$ to be isolated, see the conditions of Theorem 8 and 9 [in the notes](https://teaching.matmat.org/error-control/10_Spectra_self_adjoint.html). Considering our setting of periodic Schrödinger operators in $L^2$, look at the spectral properties of $H$ and $H_k$ argue why there is even hope that Kato-Temple can be employed in our setting?

*Note:* This you can only answer after lecture 10.
"""

# ╔═╡ 4cc3ded3-9c62-4af5-be73-7f52fa6ed177
md"""
We can not hope to use Kato temple on $H$ as $H$ has no discrete spectrum. We can use Kato temple for $H_k$ to get a reliable error bound for $\lambda_{kn}$ as a discrete spectrum, but we have to take the degeneracies into account. Also we have t o assume that we are not missing any eigenvalue $\lambda_{kn}$ so that we can have a reliable estimate for $\delta$
"""

# ╔═╡ a1c02b0d-da66-4b1a-9a4b-2a5dd7997232
md"""
**(b)** Assume that our numerical procedure (discretisation + diagonalisation) for all considered cutoffs is sufficiently good, such that no eigenpairs are missed. Employ the prevously discussed (e.g. Sheet 6) technique of combining Kato-Temple and Bauer-Fike estimates to obtain an improved estimate for the error in the eigenvalue. Add your estimate for the first eigenvalue at the $Γ$-point to your plot in Task 2.2 (d). What do you observe ?
"""

# ╔═╡ 21632ccc-690f-4fed-9091-a9444e0a79d2
for (ik, kpt) in enumerate(basis_one.kpoints)
	hamk = ham_one[ik]
	λk   = eigres_one.eigenvalues[ik]
	Xk   = eigres_one.ψ[ik]

	residual_k = hamk * Xk - Xk * Diagonal(λk)
	println(ik, "  ", norm(residual_k))
end

# ╔═╡ 7ed09f35-a22b-47c0-9ada-e3c11a9645a8
md"""
**(c)** Perform a band structure computation at cutoff $\mathcal{E} = 7$ annotated with the error estimated using the estimate developed in *(b)*. For approximate eigenvalues where you cannot obtain a Kato-Temple estimate (e.g. degeneracies), fall back to a Bauer-Fike estimate. Play with the cutoff. What do you observe ? Do the error bars correspond to the expectations of a variational convergence ? Apart from the probably unjustified assumption in *(b)*, what is the biggest drawback of the Kato-Temple estimate ?
"""

# ╔═╡ 63b1a886-6b63-4219-a941-f1a699f0d614
# Your answer here

# ╔═╡ 5f5f139e-57f9-4b2f-8398-a13fbe119fb5
md"""
-----
"""

# ╔═╡ 14df1c0f-1c4f-4da9-be49-3941b9c12fd3
md"""
### Task 2.5: A Schur-based estimate for the gap

Based on the definition of the basis cutoffs $\mathcal{E}$ and $\mathcal{F}$
and the respective projectors $P_k^\mathcal{E}$, $P_k^\mathcal{F}$ into these
bases as well as $Q_k^\mathcal{E}$ and $Q_k^\mathcal{F}$
as the projectors to the complements of the bases,
we can identify the following orthogonal decompositions of the Hilbert space
$L^2_\text{per}(\Omega)$:
```math
\begin{align}
id &= (P_k^{\mathcal{E}} + Q_k^{\mathcal{E}}) \\
   &= (P_k^{\mathcal{E}} + P_k^{\mathcal{F}} Q_k^{\mathcal{E}} + Q_k^{\mathcal{F}} Q_k^{\mathcal{E}})\\
   &= (P_k^{\mathcal{E}} + P_k^{\mathcal{R}} + Q_k^{\mathcal{F}})
\end{align}
```
where we defined $P_k^\mathcal{R} \equiv P_k^\mathcal{F} Q_k^\mathcal{E}$.
Using the notations
```math
\begin{aligned}
	H_k^{\mathcal{E}\mathcal{F}}       &= P_k^\mathcal{E} H_k P_k^\mathcal{F} &
	\quad
	H_k^{\mathcal{R}\mathcal{E}}       &= P_k^\mathcal{R} H_k P_k^\mathcal{E} \\
	H_k^{\mathcal{E}\mathcal{E}^\perp} &= P_k^\mathcal{E} H_k Q_k^\mathcal{E} &
	H_k^{\mathcal{E}\mathcal{F}^\perp} &= P_k^\mathcal{E} H_k Q_k^\mathcal{F}
\end{aligned}
```
and so on, we can decompose the Hamiltonian fiber into blocks:
```math
	H_k 
= \left(\begin{array}{ccc}
	H_k^{\mathcal{E}\mathcal{E}} & H_k^{\mathcal{E}\mathcal{E}^\perp} \\
	H_k^{\mathcal{E}^\perp\mathcal{E}} & 
	\textcolor{darkred}{H_k^{\mathcal{E}^\perp\mathcal{E}^\perp}} \\
\end{array}\right)
= 
\left(\begin{array}{ccc}
	H_k^{\mathcal{E}\mathcal{E}} & H_k^{\mathcal{E}\mathcal{R}} & H_k^{\mathcal{E}\mathcal{F}^\perp} \\
	H_k^{\mathcal{R}\mathcal{E}} & \textcolor{darkred}{H_k^{\mathcal{R}\mathcal{R}}} & \textcolor{darkred}{H_k^{\mathcal{R}\mathcal{F}^\perp}} \\
	H_k^{\mathcal{F}^\perp\mathcal{E}} & \textcolor{darkred}{H_k^{\mathcal{F}^\perp\mathcal{R}}} & \textcolor{darkred}{H_k^{\mathcal{F}^\perp\mathcal{F}^\perp}} \\
\end{array}\right).
```

We note that our iterative diagonalisation as part of the band structure calculation indeed gives us access to a few eigenpairs $(\widetilde{λ}_{kn}, \widetilde{X}_{kn})$ of $H_k^{\mathcal{E}\mathcal{E}}$. Moreover due to Courant-Fisher $λ_{kn} \leq \widetilde{λ}_{kn}$, i.e. an upper bound to $λ_{kn}$ is readily at hand. Suppose now we additionally had a lower bound $μ_{kn}$, i.e.
```math
\mu_{kn} \leq λ_{kn} \leq \widetilde{λ}_{kn}
```
then
```math
δ_{kn} \geq \min\left(μ_{kn} - \widetilde{λ}_{k,n-1}, \, μ_{k,n+1} - \widetilde{λ}_{k,n}\right) \qquad (3).
```
Finding such a suitable $μ_{kn}$ provably is the goal of this task.
"""

# ╔═╡ 3fc07beb-33c1-43b3-9d66-27693d78e46a
md"""
**(a)** Show that for Cohen-Bergstresser Hamiltonians and an appropriate choice of $\mathcal{F}$ depending on $\mathcal{E}$ (Task 2.2 (c)), the following structure of the Hamiltonian is obtained:
```math
	H_k =
\left(\begin{array}{ccc}
	H_k^{\mathcal{E}\mathcal{E}} & V_k^{\mathcal{E}\mathcal{R}} & 0\\
	V_k^{\mathcal{R}\mathcal{E}} & H_k^{\mathcal{R}\mathcal{R}} & V_k^{\mathcal{R}\mathcal{F}^\perp} \\
	0 & V_k^{\mathcal{F}^\perp\mathcal{R}} & H_k^{\mathcal{F}^\perp\mathcal{F}^\perp} \\
\end{array}\right),
```
where the use of $V$ instead of $H$ indicates that the kinetic term does not contribute to the respective block and $0$ indicates an all-zero block.
"""

# ╔═╡ 6a8ceca2-05f1-4846-b620-fe44d4c3e0d3
md"""
By 2.1, we know that if $e_G \in \mathbb{B}_k^{\mathcal{E}}$ and $|ΔG| > \sqrt{2\mathcal{E}_V}$ then
```math
\langle e_{G + ΔG}, H_k e_G \rangle = 0
```
Thus, by choosing $\mathcal{F} = \mathcal{E} + \mathcal{E}_V$, we obtain that for any $G \in \mathbb{B}_k^{\mathcal{E}}$ and $G' \notin  \mathbb{B}_k^{\mathcal{F}}$ so that $|G' - G| = |ΔG| > \sqrt{2\mathcal{E}_V}$,
```math
\langle e_{G'}, H_k e_G \rangle = 0
```
so $H_k^{\mathcal{F}^\perp \mathcal{E}} = H_k^{\mathcal{E} \mathcal{F}^\perp} = 0$. Now, for $G' \in \mathbb{B}_k^{\mathcal{F}}$ but $G' \not\in \mathbb{B}_k^{\mathcal{E}}$ so that $G \not= G'$, we have
```math
\begin{align*}
\langle e_{G'}, H_k e_G \rangle &= \langle e_{G'}, (T_k + V) e_G \rangle \\
&=\langle e_{G'}, T_k e_G \rangle + \langle e_{G'}, V e_G \rangle\\
&=\delta_{GG'} \frac12 |G+k|^2 + \langle e_{G'}, V e_G \rangle \\
&= \langle e_{G'}, V e_G \rangle
\end{align*}
```
Thus, $H_k^{\mathcal{E} \mathcal{R}} = V_k^{\mathcal{E} \mathcal{R}}$ and by symmetry $H_k^{\mathcal{R} \mathcal{E}} = V_k^{\mathcal{R} \mathcal{E}}$. A similar argument shows $H_k^{\mathcal{R} \mathcal{F}^\perp} = V_k^{\mathcal{R} \mathcal{F}^\perp}$ and by symmetry $H_k^{\mathcal{F}^\perp \mathcal{R}} = V_k^{\mathcal{F}^\perp \mathcal{R}}$.
"""

# ╔═╡ 012d944c-be57-4021-994f-eb0090b52e2b
md"----"

# ╔═╡ 047d630b-e85e-45e9-9574-758955cb160e
md"""
We focus on the splitting of $H_k$ into four blocks. An important realisation for our purpose is now that if $H_k - \mu_{kn}$ has exactly $n-1$ negative eigenvalues, then we necessarily have $\mu_{kn} < λ_{kn}$. In the following we will thus develop conditions on a parameter $\mu$, such that $H_k - \mu$ has a provable number of negative eigenvalues.

A first step is the  Haynsworth inertia additivity formula[^Hay1968]. For a matrix
```math
	H_k - \mu
= \left(\begin{array}{ccc}
	H_k^{\mathcal{E}\mathcal{E}} - \mu & V_k^{\mathcal{E}\mathcal{E}^\perp} \\
	V_k^{\mathcal{E}^\perp\mathcal{E}} & 
	H_k^{\mathcal{E}^\perp\mathcal{E}^\perp} - \mu \\
\end{array}\right).
```
this theorem states that
```math
N(H_k - \mu) = N(H_k^{\mathcal{E}\mathcal{E}} - \mu) + N(S_\mu)
```
where $N(\mathcal{A})$ denotes the number of negative eigenvalues of the operator $\mathcal{A}$ and $S_\mu$ is the Schur complement
```math
S_\mu = \left(H_k^{\mathcal{E}^\perp \mathcal{E}^\perp} - \mu\right)
- V_k^{\mathcal{E}^\perp\mathcal{E}} \left( H_k^{\mathcal{E} \mathcal{E}} - \mu \right)^{-1} V_k^{\mathcal{E}\mathcal{E}^\perp}.
```

[^Hay1968]: E. V. Haynsworth. Linear Algebra Appl. **1**, 73 (1968)
"""

# ╔═╡ 57a4ebd3-d850-4490-9992-957c15d26aea
md"""
**(b)** Using this statement explain why a $\mu \in (\widetilde{λ}_{k,n-1}, \widetilde{λ}_{kn})$ such that $S_\mu \geq 0$ is a guaranteed lower bound to $λ_{kn}$.
"""

# ╔═╡ 677a71e3-b17b-42b2-b3c7-f21cd0f81861
# Your answer here
md"""
If $\mu \in (\widetilde{\lambda}_{k, n-1}, \widetilde{\lambda}_{kn})$ such that $S_\mu \geq 0$, then $N(H_k^{\mathcal{E} \mathcal{E}} - \mu) = n-1$ since subtracting $\mu$ shifts all the eigenvalues of $H_k^{\mathcal{E} \mathcal{E}}$ down by $\mu$ so the negative eigenvalues will be $\widetilde{\lambda}_{k, 1} - \mu, \ldots, \widetilde{\lambda}_{k, n-1} - \mu$.

Since $S_\mu \geq 0$, $N(S_\mu) = 0$. Thus, by the Haynsworth interia additivity formula $N(H_k - \mu) = n-1$ and so $\mu > \lambda_{kn}$.
"""

# ╔═╡ a70934fa-e78f-4592-90f8-57af5e9cb87a
md"""
**(c)** We proceed to obtain conditions that ensure $S_\mu \geq 0$. Let $X = L^2_\text{per}(\Omega) \backslash \text{span}\left(\mathbb{B}_k^\mathcal{E}\right)$. Recall that $S_\mu \geq 0$ exactly if
```math
\langle x,  S_\mu x \rangle \geq 0 \qquad \forall x \in X
```
Prove the following statements:
- First $\forall x \in X$
  ```math
  \left\langle x \,\middle|\, (H_k^{\mathcal{E}^\perp\mathcal{E}^\perp} - \mu) x \right\rangle
  \geq \mathcal{E} - \|V_k^{\mathcal{E}^\perp\mathcal{E}^\perp} \|_\text{op} - \mu,
  ```
  where $\|\,\cdot\,\|_\text{op}$ is the standard Hilbert operator norm
  ```math
  \|\mathcal{A}\|_\text{op} = \sup_{0\neq\varphi\in L^2_\text{per}(\Omega)} \frac{\langle\varphi, \mathcal{A} \varphi\rangle}{\langle\varphi, \varphi\rangle}
  ```
- Second, given a full eigendecomposition $H_k^{\mathcal{E} \mathcal{E}} = \widetilde{X}_k \widetilde{Λ}_k \widetilde{X}_k^H$ show that
  for $\mu \in I_n$ with the open interval
  ```math
  I_n = \left(\frac12 \left(\widetilde{λ}_{k,n-1} + \widetilde{λ}_{kn}\right), \, \widetilde{λ}_{kn}\right)
  ```
  we have $\forall x \in X$:
  ```math
  \begin{align}
  \left\langle x \, \middle|\,
  V_k^{\mathcal{E}^\perp\mathcal{E}} \left( H_k^{\mathcal{E} \mathcal{E}} - \mu   \right)^{-1} V_k^{\mathcal{E}\mathcal{E}^\perp} x \right\rangle
  &\leq \left\|\left(V_k^{\mathcal{R}\mathcal{E}} \widetilde{X}_k\right) \left(\widetilde{Λ}_k - \mu\right)^{-1} \left( V_k^{\mathcal{R}\mathcal{E}} \widetilde{X}_k \right)^H \right\|_\text{op}\\
  &\leq \frac{\|V_k\|_\text{op}^2}{\widetilde{λ}_{kn} - \mu}
  \end{align}
  ```
- Combining both show that
  ```math
  S_\mu \geq \mathcal{E} - \mu - l_1^V - \frac{(l_1^V)^2}{\widetilde{λ}_{kn} - \mu} 
  \quad \text{where} \quad l_1^V = \sum_{|G| < \sqrt{2\mathcal{E}_V}} |\hat{V}(G)|. \quad\quad (4)
  ```
  You may want to use Young's inequality, that is
  ```math
  \|V_k\|_\text{op} \leq \sum_{G \in \mathbb{L}^\ast} |\hat{V}(G)|.
  ```
"""

# ╔═╡ 3fd29e37-d6ea-427a-9b25-2e1dcba42ed0
# Your answer here
md"""
(1) Since $X = \text{span}\{e_G \, \Big| \, \frac{1}{2}|G + k|^2 \geq  \mathcal{E}\}$, it suffices to prove the first inequality for $x = e_G$ where $\frac{1}{2}|G + k|^2 \geq  \mathcal{E}$. 
```math
\begin{align*}
\langle e_G | (H_k^{\mathcal{E}^\perp \mathcal{E}^\perp} - \mu) e_G \rangle &= \langle e_G | Q_k^{\mathcal{E}^\perp}(T_k + V) e_G \rangle - \mu \\
&= \langle e_G | \frac{1}{2} |G + k|^2 e_G \rangle + \langle e_G | Q_k^{\mathcal{E}^\perp} V e_G\rangle - \mu \\
&\geq \mathcal{E} - \|V_k^{\mathcal{E}^\perp\mathcal{E}^\perp} \|_\text{op} - \mu
\end{align*}
```
where $\langle e_G | Q_k^{\mathcal{E}^\perp} V e_G\rangle \geq - \|V_k^{\mathcal{E}^\perp\mathcal{E}^\perp} \|_\text{op}$ follows from definition.
"""

# ╔═╡ d5de7b64-db19-48cf-81e4-d1f48c6ed08b
md"""
(2) Again, it suffices to prove the inequality for $x = e_G$ where $\frac{1}{2} | G + k|^2 \geq \mathcal{E}$.
```math
\begin{align}
\left\langle e_G \, \middle|\,
  V_k^{\mathcal{E}^\perp\mathcal{E}} \left( H_k^{\mathcal{E} \mathcal{E}} - \mu   \right)^{-1} V_k^{\mathcal{E}\mathcal{E}^\perp} e_G \right\rangle &\leq \| V_k^{\mathcal{E}^\perp\mathcal{E}} \left( H_k^{\mathcal{E} \mathcal{E}} - \mu   \right)^{-1} V_k^{\mathcal{E}\mathcal{E}^\perp} \|_{\text{op}} \\
&= \| V_k^{\mathcal{E}^\perp\mathcal{E}} \left( \widetilde{X}_k \widetilde{Λ}_k \widetilde{X}_k^H - \mu   \right)^{-1} V_k^{\mathcal{E}\mathcal{E}^\perp} \|_{\text{op}} \\
&= \| V_k^{\mathcal{E}^\perp\mathcal{E}} \widetilde{X}_k \left( \widetilde{Λ}_k - \mu \right)^{-1} \widetilde{X}_k^H V_k^{\mathcal{E}\mathcal{E}^\perp} \|_{\text{op}} \\
&= \left\| \left( V_k^{\mathcal{R}\mathcal{E}} \widetilde{X}_k \right) \left( \widetilde{Λ}_k - \mu \right)^{-1} \left( V_k^{\mathcal{R}\mathcal{E}} \widetilde{X}_k \right)^H \right\|_{\text{op}} \\
&\leq \left\|V_k^{\mathcal{R}\mathcal{E}} \widetilde{X}_k \right\|_{\text{op}}^2 \left\| \left(\widetilde{Λ}_k - \mu \right)^{-1} \right\|_{\text{op}} \\
&\leq \frac{\| V_k \|^2_{\text{op}}}{\min |\widetilde{λ}_{kj} - \mu|} \\
&= \frac{\| V_k \|^2_{\text{op}}}{\widetilde{λ}_{kn} - \mu}
\end{align}
```
"""

# ╔═╡ e2ff4156-5894-4ed6-a52e-6909c5f03bf8
md"""
(3) Now we combine both inequalities to derive
"""

# ╔═╡ 5bfb143e-414b-4283-9a6a-be99452aa1b5
md"----"

# ╔═╡ d6e7e0d0-52d8-4306-bde0-1a55598ed6af
md"""
The strategy to determine $\mu$ is thus as follows:

1. Compute $l_1^V$. For this we need a Fourier representation of $V$. A real space representation is available as `Vreal = DFTK.total_local_potential(ham)` where `ham` is a `DFTK.Hamiltonian`, which can be transformed into Fourier space using a fast-fourier transform `fft(basis, kpoint, Vreal)` where `basis` is the `ham.basis` and `kpoint` is the `Kpoint` object matching the currently processed $k$. Note that this assumes that $\mathcal{E}_V < \mathcal{E}$.

2. Using the lower bound $(4)$ find the largest $\mu \in I_n$ for which it can be ensured that $S_\mu \geq 0$. Since we are not using the exact $S_\mu$ here, but only a lower bound, this step can fail (i.e. both possible solutions $\mu$ are outside of $I_n$). In this case we are unable to obtain a lower bound for $\widetilde{λ}_{kn}$ and thus unable to obtain a Kato-Temple estimate.

3. If 2. goes through we set $\mu_{kn} = \mu$ and thus have guaranteed bounds $\mu_{kn} \leq λ_{kn} \leq \widetilde{λ}_{kn}$.
"""

# ╔═╡ d4792fec-7762-4c2b-9be8-3c6c6791bd3d
md"""
**(d)** Focusing on the first two eigenpairs of the $\Gamma$-point of the Cohen-Bergstresser model, plot both the guaranteed lower bound $\mu_{kn}$ from above procedure as well as the non-guarandeed bound of Task 2.4 as you increase $\mathcal{E}$. Take the computation of $\widetilde{λ}_{nk}$ for $\mathcal{E} = 80$ as the reference. Which bound is sharper ? While it should be emphasised that better guaranteed bounds than our development are possible (see for example [^HLC2020]), can you comment on the advantages and disadvantages of guaranteed error bounds ?
"""

# ╔═╡ 10426573-4929-4576-9633-781b024bd97b
# Your answer here

# ╔═╡ 3de33cb8-c6ee-4b60-94df-2cf29afbe2d7
md"""
--------

[^HLC2020]:  M. F. Herbst, A. Levitt and E. Cancès. Faraday Discuss., **224**, 227 (2020). DOI [10.1039/D0FD00048E](https://doi.org/10.1039/D0FD00048E)
"""

# ╔═╡ 2c6f639e-fbf2-4d9a-a0dd-4306f0b92a8d
md"""
### Task 2.6: Band structure with guaranteed bounds

**(a)**
Based on the results so far compute a band structure with guaranteed error bars for $\mathcal{E} = 7$ and the $6$ first bands (i.e. the $6$ lowest eigenpairs). For each eigenvalue:
- Estimate the **algorithm** and **arithmetic error** by re-computing the in-basis residual $P_k^\mathcal{E}r_{kn}$ using `Double64`. For this you need to build a basis and a Hamiltonian that employes `Double64` as the working precision. To change DFTK's internal working precision follow the [Arbitrary floating-point types](https://docs.dftk.org/stable/examples/arbitrary_floattype/) documentation page. From a `basis_double64` using `Double64` precision a corresponding Hamiltonian is obtained by `Hamiltonian(basis_double64)` as shown before. *Hint:* Before computing products `Hamiltonian(basis_double64) * X`, ensure that `X` has been converted to `Double64` as well.
- Estimate the **discretisation error** using both the Bauer-Fike estimates of Tasks 2.2 & 2.3 as well as the Kato-Temple estimates of Task 2.5. In the band structure annotate the tightest bound that is available to you.

"""

# ╔═╡ f0b9d21b-dc78-4d95-ac47-5a19f41c0fd4
# Your answer here

# ╔═╡ 6a83edeb-fe78-4ad5-98a2-d1ef33c333e4
md"""
------
"""

# ╔═╡ 33896109-d190-4992-806a-c447ca36071b
md"""
## Adaptive diagonalisation techniques

The main purpose of *a posteriori* error bounds as we have developed in the previous tasks is to understand the  error in the current computational setup. A good error bound is sharp, cheap to compute and ideally guaranteed, i.e. the actual error is always smaller. One can thus think of the error bars as safety checks. If one is not yet satisfied with the current quality of results, one can always choose a more accurate numerical setup and re-compute. For example if the estimated discretisation error is too large, one just increases $\mathcal{E}$ until one is satisfied.

As simple as this procedure is, it has one notable drawback: It leads to a nested set of loops, where for each attempted $\mathcal{E}$ (outer loop), we iteratively diagonalise the $H_k$ using LOBPCG (inner loop). A natural question is thus, whether one can fuse these two loops, i.e. adapt $\mathcal{E}$ *while the LOBPCG is converging*, such that at each step the algorithm error and discretisation error are balanced.

To make this idea more clear, let us recall a basic LOBPCG implementation, which in this case only performs a diagonalisation of the Hamiltonian fiber at the gamma point of a passed model:
"""

# ╔═╡ 93552ccd-f7b5-4830-a9ad-417d3fca9af9
function nonadaptive_lobpcg(model::Model{T}, Ecut, n_bands;
                            maxiter=100, tol=1e-6, verbose=false) where {T}
	kgrid = (1, 1, 1)  # Γ point only
	basis = PlaneWaveBasis(model; Ecut, kgrid)
	ham   = Hamiltonian(basis)
	hamk  = ham[1]                   # Select Γ point
	prec  = PreconditionerTPA(hamk)  # Initialise preconditioner
	X     = DFTK.random_orbitals(hamk, n_bands)  # Random initial guess
	
	converged = false
	λ = NaN
	residual_norms = NaN
	residual_history = []
	
	P = zero(X)
	R = zero(X)
	for i in 1:maxiter
		if i > 1
			Z = hcat(X, P, R)
		else
			Z = X
		end
		Z = Matrix(qr(Z).Q)  # QR-based orthogonalisation

		# Rayleigh-Ritz
		HZ = hamk * Z
		λ, Y = eigen(Hermitian(Z' * HZ))
		λ = λ[1:n_bands]
		Y = Y[:, 1:n_bands]
		new_X = Z * Y
		
		# Compute residuals and convergence check
		R = HZ * Y - new_X * Diagonal(λ)
		residual_norms = norm.(eachcol(R))
		push!(residual_history, residual_norms)
		verbose && @printf "%3i %8.4g %8.4g\n" i λ[end] residual_norms[end]
		if maximum(residual_norms) < tol
			converged = true
			X .= new_X
			break
		end

		# Precondition and update
		DFTK.precondprep!(prec, X)
		ldiv!(prec, R)
		P .= X - new_X
		X .= new_X

		# Additional step:
		# Move to larger basis ?
	end

	# Notice that eigenvalues are returned as λ
	# and eigenvectors as X
	(; λ, X, basis, ham, converged, residual_norms, residual_history)
end

# ╔═╡ 968a26f2-12fe-446f-aa41-977fdffc23a0
md"""
This implementation can be easily compared to the `DFTK.lobpcg_hyper` function of DFTK. For example:
"""

# ╔═╡ 7ebc18b0-618f-4d99-881f-7c30eb3bc7f5
let
	Ecut     = 10
	n_bands  = 6

	# Run the nonadaptive function above. Returns eigenvalues as
	# res_nonadaptive.λ and eigenvectors as res_nonadaptive.X
	res_nonadaptive = nonadaptive_lobpcg(model, Ecut, n_bands)

	# Setup discretisation
	basis = PlaneWaveBasis(model; Ecut=10, kgrid=(1, 1, 1))
	ham   = Hamiltonian(basis)

	hamk  = ham[1]  # Select Γ point
	prec  = PreconditionerTPA(hamk)  # Initialise preconditioner
	X = DFTK.random_orbitals(hamk, n_bands)  # Random initial guess

	# Run eigensolver from DFTK, also returns eigenvalues
	# as res_dftk.λ and eigenvectors as res_dftk.X
	res_dftk = DFTK.lobpcg_hyper(hamk, X; prec, tol=1e-6)

	# Take the difference between eigenvalues
	res_nonadaptive.λ - res_dftk.λ
end

# ╔═╡ bf00ec9d-5c69-43c3-87c1-28ac9f5b4c9f
md"""
In order to make this routine discretisation-adaptive we need to add an additional step as indicated in the above source code: After we have checked for convergence and as we are updating `X`, `P` and `R` to their new values, we might additionally want to increase the discretisation basis to employ a new, increased cutoff value $\mathcal{E} + \Delta$ to discretise the Hamiltonian. Using
```julia
DFTK.transfer_blochwave_kpt(X, basis_small, basis_small.kpoints[1],
                               basis_large, basis_large.kpoints[1])
```
we can then transfer the state vectors `X`, `P` and `R` to this basis.

This means while the iterative eigensolver is converging we also improve the discretisation basis by successively increasing $\mathcal{E}$. Each LOBPCG iteration may thus run using a different discretisation basis.
"""

# ╔═╡ 9430a871-0f8f-4ee2-b574-f818f1580abd
md"""
### Task 2.7: Developing a discretisation-adaptive LOBPCG

Assume the LOBPCG currently runs at cutoff $\mathcal{F}$. Due to the properties of the Cohen-Bergstresser Hamiltonian (Task 2.2 (b)) there is a smaller cutoff $\mathcal{E} < \mathcal{F}$ such that (Task 2.5 (a))
```math
Q_k^{\mathcal{F}} H P_k^{\mathcal{E}} = 0.
```

**(a)** Prove that if $\|Q_k^{\mathcal{E}} \widetilde{X}_{kn}\| = 0$, then
```math
|λ_{kn} - \widetilde{λ}_{kn}| \leq \|P_k^{\mathcal{F}} r_{kn}\|,
```
i.e. that the total error is driven to zero as the iterative diagonalisation employing a cutoff $\mathcal{F}$ is converging. Based on this argue why a large value for $\|Q_k^{\mathcal{E}} \widetilde{X}_{kn}\|$ relative to $\|P_k^{\mathcal{F}} r_{kn}\|$ provides a reasonable indicator to decide when to refine the discretisation.
"""

# ╔═╡ 3c1c0131-96fe-4fb7-ac0c-0aaec21dd023
# Your answer here

# ╔═╡ 5de67497-d252-443a-8a72-8e01891aea41
md"""
**(b)** Following the idea of (a) code up an adaptive LOBPCG by monitoring the ratio
```math
\frac{\|Q_k^{\mathcal{E}} \widetilde{X}_{kn}\|}
{\|P_k^{\mathcal{F}} r_{kn}\|}  \qquad \quad (5).
```
Note, that one way to compute $Q_k^{\mathcal{E}} \widetilde{X}_{kn}$ is as $\widetilde{X}_{kn} - P_k^{\mathcal{E}} \widetilde{X}_{kn}$ where $P_k^{\mathcal{E}}$ can be obtained by transferring from the large basis to the small basis and back to the large basis, i.e.
```julia
X_small = transfer_blochwave(X_large,
						   basis_large, basis_large.kpoints[1],
						   basis_small, basis_small.kpoints[1])
P_X     = transfer_blochwave(X_small,
						   basis_small, basis_small.kpoints[1],
						   basis_large, basis_large.kpoints[1])
```
Whenever (5) becomes too large, you should switch to a finer discretisation basis, e.g. by switching from a cutoff $\mathcal{E}$ to $\mathcal{E} + \Delta$. Experiment with this setup to find what are good ratios $(5)$ to indicate switching and good values for $\Delta$. Some strategy for exploration:
- Assume we eventually want to target a cutoff $\mathcal{F}^\text{final} = 80$. Save the convergence history for running the full LOBPCG at exactly this cutoff. This is your reference.
- With your adaptive methods you should try to loose as little as possible in the rate of convergence compared to the reference convergence profile, but try to use as small values of $\mathcal{F}$ for as many iterations as you can.
- To develop your approach, first only compute a single eigenvalue (`n_bands = 1`) and then start considering more than one. Note, that you will need to adapt $(5)$ and take appropriate maxima / minima over all computed eigenpairs to decide when to switch to the next bigger basis.
- Be creative and experiment. In this Task there is no "best" solution.
"""

# ╔═╡ 8c712ee5-f076-442e-ad33-edfe5a6e6941
# Your answer here

# ╔═╡ ee3fe129-48f8-4932-ac23-2287c6be9780
md"""
## Part 3: Conclusion

To conclude the project, we will zoom out a little from the concrete tasks.
"""

# ╔═╡ b61a9e8c-e7b4-4e5e-8b4a-10e939c164a9
md"""
### Task 3.1 Summary and takaways from the project

Write a short summary of the main outcomes of your project (about one A4 page, can be submitted as a separate PDF).

In your summary you should answer the following questions, making reference to the ideas and examples you explored in this project:

- Provide the key ingredients to effiently find the lowermost eigenvalues in the spectrum of an operator. How can error estimation help ? How can thinking about errors help ?
- Why can it be useful to introduce certain errors (model, discretisation, floating-point) *on purpose* ?
- Considering the techniques developed in Part 1 for finite-difference discretisations and the ones of Part 2 for plane-wave discretisations, can you make a guess how easy it would be to apply the methods in respectively the context of the other discretisation ?
- Given the effort required to arrive at these results, are you satisfied with the outcome (think about efficiency improvements, sharpness of estimates, etc.)
- Overall, what is your key achievement in Part 1, what is your key achievement in Part 2 ?
"""

# ╔═╡ ea9b6c1b-e2dd-4c68-82ca-b63ada51cfd1
md"""
### Task 3.2: Group atmosphere and distribution of work

**(a)** Each team member individually should provide 4-5 sentences on his role in the project and the overall experience of working on the project as a group. We will open a separate task on moodle for you to return this feedback there, if you prefer.

Some questions you should address:
- Which tasks and subtasks (*(a)*, *(b)*, etc.) of the project did you mostly work on ?
- How did you decide within the group to distribute the workload as such ?
- In your opinion did each group member contribute equally to the project ?
- Where could your specific expertise and background from your prior studies contribute most to the project and the exercises we did earlier in the semester?
- Can you pinpoint aspects about your team members' study subject, which are relevant to the course (either the exercises or this project), which you learned from them in your discussions ?
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Brillouin = "23470ee3-d0df-4052-8b1a-8cbd6363e7f0"
DFTK = "acf6eb54-70d9-11e9-0013-234b7a5f5337"
DoubleFloats = "497a8b3b-efae-58df-a0af-a86822472b78"
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
LimitedLDLFactorizations = "f5a24dde-3ab7-510b-b81b-6a72c6098d3b"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LinearMaps = "7a12625a-238d-50fd-b39a-03d52299707e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Preconditioners = "af69fa37-3177-5a40-98ee-561f696e4fcd"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[compat]
BenchmarkTools = "~1.5.0"
Brillouin = "~0.5.19"
DFTK = "~0.6.20"
DoubleFloats = "~1.4.0"
Interpolations = "~0.15.1"
LimitedLDLFactorizations = "~0.5.1"
LinearMaps = "~3.11.3"
Plots = "~1.40.5"
PlutoTeachingTools = "~0.3.1"
PlutoUI = "~0.7.59"
Preconditioners = "~0.6.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.2"
manifest_format = "2.0"
project_hash = "a02cbb36819f7abfc48d4e21cc63858f44394183"

[[deps.AMD]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse_jll"]
git-tree-sha1 = "45a1272e3f809d36431e57ab22703c6896b8908f"
uuid = "14f7f29c-3bd6-536c-9a0b-7339e30b5a3e"
version = "0.5.3"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown"]
git-tree-sha1 = "b392ede862e506d451fc1616e79aa6f4c673dab8"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.38"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsDatesExt = "Dates"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsTestExt = "Test"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AlgebraicMultigrid]]
deps = ["CommonSolve", "LinearAlgebra", "Printf", "Reexport", "SparseArrays"]
git-tree-sha1 = "796eedcb42226861a51d92d28ee82d4985ee860b"
uuid = "2169fc97-5a83-5252-b627-83903c6c433c"
version = "0.5.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "3640d077b6dafd64ceb8fd5c1ec76f7ca53bcf76"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.16.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.AtomsBase]]
deps = ["LinearAlgebra", "PeriodicTable", "Printf", "Requires", "StaticArrays", "Unitful", "UnitfulAtomic"]
git-tree-sha1 = "995c2b6b17840cd87b722ce9c6cdd72f47bab545"
uuid = "a963bdd2-2df7-4f54-a1ee-49d51e6be12a"
version = "0.3.5"

[[deps.AtomsCalculators]]
deps = ["AtomsBase", "StaticArrays", "Test", "Unitful"]
git-tree-sha1 = "3135b34463e2df7105e31fced8415bffb18704f7"
uuid = "a3e0e189-c65a-42c1-833c-339540406eb1"
version = "0.2.2"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "f1dff6729bc61f4d49e140da1af55dcd1ac97b2f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.5.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bravais]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "55f9f5585b95bcbc206e623fbe5a7ffb8eb1fe34"
uuid = "ada6cbde-b013-4edf-aa94-f6abe8bd6e6b"
version = "0.2.2"

[[deps.Brillouin]]
deps = ["Bravais", "DirectQhull", "DocStringExtensions", "LinearAlgebra", "PrecompileTools", "Reexport", "Requires", "StaticArrays"]
git-tree-sha1 = "f24f75919cd037923f4fa77aae13611e8dae100a"
uuid = "23470ee3-d0df-4052-8b1a-8cbd6363e7f0"
version = "0.5.19"

    [deps.Brillouin.extensions]
    BrillouinMakieExt = "Makie"
    BrillouinPlotlyJSExt = "PlotlyJS"
    BrillouinSpglibExt = "Spglib"

    [deps.Brillouin.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    PlotlyJS = "f0f68f2c-4968-5e81-91da-67840de0976a"
    Spglib = "f761d5c5-86db-4880-b97f-9680a7cccfb5"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8873e196c2eb87962a2048b3b8e08946535864a1"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+2"

[[deps.CUDA_Driver_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ccd1e54610c222fadfd4737dac66bff786f63656"
uuid = "4ee394cb-3365-5eb0-8335-949819d2adfc"
version = "0.10.3+0"

[[deps.CUDA_Runtime_jll]]
deps = ["Artifacts", "CUDA_Driver_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "TOML"]
git-tree-sha1 = "e43727b237b2879a34391eeb81887699a26f8f2f"
uuid = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
version = "0.15.3+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "3e4b134270b372f2ed4d4d0e936aabaefc1802bc"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "13951eb68769ad1cd460cdb2e64e5e95f1bf123d"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "362a287c3aa50601b0bc359053d5c2468f0e7ce0"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.11"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ComponentArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "ForwardDiff", "Functors", "LinearAlgebra", "PackageExtensionCompat", "StaticArrayInterface", "StaticArraysCore"]
git-tree-sha1 = "bc391f0c19fa242fb6f71794b949e256cfa3772c"
uuid = "b0b7db55-cfe3-40fc-9ded-d10e2dbeff66"
version = "0.15.17"

    [deps.ComponentArrays.extensions]
    ComponentArraysAdaptExt = "Adapt"
    ComponentArraysConstructionBaseExt = "ConstructionBase"
    ComponentArraysGPUArraysExt = "GPUArrays"
    ComponentArraysOptimisersExt = "Optimisers"
    ComponentArraysRecursiveArrayToolsExt = "RecursiveArrayTools"
    ComponentArraysReverseDiffExt = "ReverseDiff"
    ComponentArraysSciMLBaseExt = "SciMLBase"
    ComponentArraysTrackerExt = "Tracker"
    ComponentArraysTruncatedStacktracesExt = "TruncatedStacktraces"
    ComponentArraysZygoteExt = "Zygote"

    [deps.ComponentArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    Optimisers = "3bd65402-5787-11e9-1adc-39752487f4e2"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SciMLBase = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    TruncatedStacktraces = "781d530d-4396-4725-bb49-402e4bee1e77"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "ea32b83ca4fefa1768dc84e504cc0a94fb1ab8d1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CrystallographyCore]]
deps = ["LinearAlgebra", "StaticArrays", "StructEquality"]
git-tree-sha1 = "521a8ed44455592672d887632eef056c1ba4678c"
uuid = "80545937-1184-4bc9-b283-396e91386b5c"
version = "0.3.3"

[[deps.DFTK]]
deps = ["AbstractFFTs", "Artifacts", "AtomsBase", "AtomsCalculators", "Brillouin", "ChainRulesCore", "Dates", "DftFunctionals", "DocStringExtensions", "FFTW", "ForwardDiff", "GPUArraysCore", "Interpolations", "IterTools", "IterativeSolvers", "LazyArtifacts", "Libxc", "LineSearches", "LinearAlgebra", "LinearMaps", "MPI", "Markdown", "Optim", "PeriodicTable", "PkgVersion", "Polynomials", "PrecompileTools", "Preferences", "Primes", "Printf", "PseudoPotentialIO", "Random", "Roots", "SparseArrays", "SpecialFunctions", "Spglib", "StaticArrays", "Statistics", "TimerOutputs", "Unitful", "UnitfulAtomic"]
git-tree-sha1 = "99b5bb711d2b4e5b7cd6d94827e6878edf51085b"
uuid = "acf6eb54-70d9-11e9-0013-234b7a5f5337"
version = "0.6.20"

    [deps.DFTK.extensions]
    DFTKCUDAExt = "CUDA"
    DFTKGenericLinearAlgebraExt = "GenericLinearAlgebra"
    DFTKIntervalArithmeticExt = "IntervalArithmetic"
    DFTKJLD2Ext = "JLD2"
    DFTKJSON3Ext = "JSON3"
    DFTKPlotsExt = "Plots"
    DFTKWannier90Ext = "wannier90_jll"
    DFTKWannierExt = "Wannier"
    DFTKWriteVTKExt = "WriteVTK"

    [deps.DFTK.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GenericLinearAlgebra = "14197337-ba66-59df-a3e3-ca00e7dcff7a"
    IntervalArithmetic = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
    JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
    JSON3 = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
    Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
    Wannier = "2b19380a-1f7e-4d7d-b1b8-8aa60b3321c9"
    WriteVTK = "64499a7a-5c06-52f2-abe2-ccb03c286192"
    wannier90_jll = "c5400fa0-8d08-52c2-913f-1e3f656c1ce9"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DftFunctionals]]
deps = ["ComponentArrays", "DiffResults", "ForwardDiff"]
git-tree-sha1 = "0ae7d5b8d65091e4b59f5859c85be4427c57de11"
uuid = "6bd331d2-b28d-4fd3-880e-1a1c7f37947f"
version = "0.2.4"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DirectQhull]]
deps = ["Qhull_jll"]
git-tree-sha1 = "49951cee00be263d4dc187f38aa96333d74f4d1c"
uuid = "c3f9d41a-afcb-471e-bc58-0b8d83bd86f4"
version = "0.2.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.DoubleFloats]]
deps = ["GenericLinearAlgebra", "LinearAlgebra", "Polynomials", "Printf", "Quadmath", "Random", "Requires", "SpecialFunctions"]
git-tree-sha1 = "1c4a523eaaea709f77e33d973466437af12dfc59"
uuid = "497a8b3b-efae-58df-a0af-a86822472b78"
version = "1.4.1"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c6317308b9dc757616f0b5cb379db10494443a7"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.2+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.EzXML]]
deps = ["Printf", "XML2_jll"]
git-tree-sha1 = "380053d61bb9064d6aa4a9777413b40429c79901"
uuid = "8f5d6c58-4d21-5cfd-889c-e3ad7ee6a615"
version = "1.2.0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+1"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

    [deps.FillArrays.weakdeps]
    PDMats = "90014a1f-27ba-587c-ab20-58faa44d9150"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "b10bdafd1647f57ace3885143936749d61638c3b"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.26.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "5c1d8ae0efc6c2e7b1fc502cbe25def8f661b7bc"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.2+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.Functors]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "64d8e93700c7a3f28f717d265382d52fac9fa1c1"
uuid = "d9f16b24-f501-4c13-a1f2-28368ffc5196"
version = "0.4.12"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "532f9126ad901533af1d4f5c198867227a7bb077"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+1"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "ec632f177c0d990e64d955ccc1b8c04c485a0950"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.6"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "ee28ddcd5517d54e417182fec3886e7412d3926f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.8"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f31929b9e67066bee48eec8b03c0df47d31a74b3"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.8+0"

[[deps.GenericLinearAlgebra]]
deps = ["LinearAlgebra", "Printf", "Random", "libblastrampoline_jll"]
git-tree-sha1 = "c4f9c87b74aedf20920034bd4db81d0bffc527d2"
uuid = "14197337-ba66-59df-a3e3-ca00e7dcff7a"
version = "0.3.14"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "674ff0db93fffcd11a3573986e550d66cd4fd71f"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.80.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "bc3f416a965ae61968c20d0ad867556367f2817d"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.9"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "401e4f3f30f43af2c8478fc008da50096ea5240f"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.3.1+0"

[[deps.Hwloc_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dd3b49277ec2bb2c6b94eb1604d4d0616016f7a6"
uuid = "e33a78d0-f292-5ffc-b300-72abe9b543c8"
version = "2.11.2+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "b8ffb903da9f7b8cf695a8bead8e01814aa24b30"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.2"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "10bd689145d2c3b2a9844005d01087cc1194e79e"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.1+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "59545b0a2b27208b0650df0a46b8e3019f85055b"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.4"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "39d64b09147620f5ffbf6b2d3255be3c901bec63"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.8"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "be3dc50a92e5a386872a493a10050136d4703f9b"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "25ee0be4d43d0269027024d75a24c24d6c6e590c"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.4+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "2984284a8abcfcc4784d95a9e2ea4e352dd8ede7"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.36"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "36bdbc52f13a7d1dcb0f3cd694e01677a515655b"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "854a9c268c43b77b0a27f22d7fab8d33cdb3a731"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+1"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "ce5f5621cac23a86011836badfedf664a612cee4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.5"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c6ce1e19f3aec9b59186bdf06cdf3c4fc5f5f3e6"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.50.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "61dfdba58e585066d8bce214c5a51eaa0539f269"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0c4f9c4f1a50d8f35048fa0532dabbadf702f81e"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.1+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "b404131d06f7886402758c9ce2214b636eb4d54a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5ee6203157c120d79034c748a2acba45b82b8807"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.1+0"

[[deps.Libxc]]
deps = ["Libxc_GPU_jll", "Libxc_jll", "Requires"]
git-tree-sha1 = "a8af8c180bbd455aed17a950f2efa86f8b1d10bf"
uuid = "66e17ffc-8502-11e9-23b5-c9248d0eb96d"
version = "0.3.18"

    [deps.Libxc.extensions]
    LibxcCudaExt = "CUDA"

    [deps.Libxc.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[[deps.Libxc_GPU_jll]]
deps = ["Artifacts", "CUDA_Runtime_jll", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "Pkg", "TOML"]
git-tree-sha1 = "ee321f68686361802f2ddb978dae441a024e61ea"
uuid = "25af9330-9b41-55d4-a324-1a83c0a0a1ac"
version = "6.1.0+2"

[[deps.Libxc_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c5516f2b1655a103225e69477e3df009347580df"
uuid = "a56a6d9d-ad03-58af-ab61-878bf78270d6"
version = "6.1.0+0"

[[deps.LimitedLDLFactorizations]]
deps = ["AMD", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "07b3c94ba4decdd855770beddeab9187be03d6b8"
uuid = "f5a24dde-3ab7-510b-b81b-6a72c6098d3b"
version = "0.5.1"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "e4c3be53733db1051cc15ecf573b1042b3a712a1"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.3.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LinearMaps]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ee79c3208e55786de58f8dcccca098ced79f743f"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.11.3"
weakdeps = ["ChainRulesCore", "SparseArrays", "Statistics"]

    [deps.LinearMaps.extensions]
    LinearMapsChainRulesCoreExt = "ChainRulesCore"
    LinearMapsSparseArraysExt = "SparseArrays"
    LinearMapsStatisticsExt = "Statistics"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "260dc274c1bc2cb839e758588c63d9c8b5e639d1"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.0.5"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MPI]]
deps = ["Distributed", "DocStringExtensions", "Libdl", "MPICH_jll", "MPIPreferences", "MPItrampoline_jll", "MicrosoftMPI_jll", "OpenMPI_jll", "PkgVersion", "PrecompileTools", "Requires", "Serialization", "Sockets"]
git-tree-sha1 = "892676019c58f34e38743bc989b0eca5bce5edc5"
uuid = "da04e1cc-30fd-572f-bb4f-1f8673147195"
version = "0.20.22"

    [deps.MPI.extensions]
    AMDGPUExt = "AMDGPU"
    CUDAExt = "CUDA"

    [deps.MPI.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"

[[deps.MPICH_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "7715e65c47ba3941c502bffb7f266a41a7f54423"
uuid = "7cb0a576-ebde-5e09-9194-50597f1243b4"
version = "4.2.3+0"

[[deps.MPIPreferences]]
deps = ["Libdl", "Preferences"]
git-tree-sha1 = "c105fe467859e7f6e9a852cb15cb4301126fac07"
uuid = "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267"
version = "0.1.11"

[[deps.MPItrampoline_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML"]
git-tree-sha1 = "70e830dab5d0775183c99fc75e4c24c614ed7142"
uuid = "f1f71cc9-e9ae-5b93-9b94-4fe0e1ad3748"
version = "5.5.1+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicrosoftMPI_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f12a29c4400ba812841c6ace3f4efbb6dbb3ba01"
uuid = "9237b28f-5490-5468-be7b-bb81f5f5e6cf"
version = "10.1.4+2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OffsetArrays]]
git-tree-sha1 = "1a27764e945a152f7ca7efa04de513d473e9542e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.1"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenMPI_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Hwloc_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "MPIPreferences", "TOML", "Zlib_jll"]
git-tree-sha1 = "bfce6d523861a6c562721b262c0d1aaeead2647f"
uuid = "fe0851c0-eecd-5654-98d4-656369965a5c"
version = "5.0.5+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7493f61f55a6cce7325f197443aa80d32554ba10"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "d9b79c4eed437421ac4285148fcadf42e0700e89"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.9.4"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"
weakdeps = ["Requires", "TOML"]

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e127b609fb9ecba6f201ba7ab753d5a605d53801"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.54.1+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.PeriodicTable]]
deps = ["Base64", "Unitful"]
git-tree-sha1 = "238aa6298007565529f911b734e18addd56985e1"
uuid = "7b2266bf-644c-5ea3-82d8-af4bbd25a884"
version = "1.2.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "650a022b2ce86c7dcfbdecf00f78afeeb20e5655"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.2"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "45470145863035bb124ca51b320ed35d071cc6c2"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.8"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoLinks", "PlutoUI"]
git-tree-sha1 = "8252b5de1f81dc103eb0293523ddf917695adea1"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.3.1"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "1a9cfb2dc2c2f1bd63f1906d72af39a79b49b736"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.11"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preconditioners]]
deps = ["AlgebraicMultigrid", "LimitedLDLFactorizations", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "a02fc23058e6264b0f73523e99e2dd1a20232dff"
uuid = "af69fa37-3177-5a40-98ee-561f696e4fcd"
version = "0.6.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "cb420f77dc474d23ee47ca8d14c90810cafe69e7"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.6"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.PseudoPotentialIO]]
deps = ["EzXML", "LinearAlgebra"]
git-tree-sha1 = "88cf9598d70015889c99920ff3dacca0eb26ae90"
uuid = "cb339c56-07fa-4cb2-923a-142469552264"
version = "0.1.1"

[[deps.Qhull_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be2449911f4d6cfddacdf7efc895eceda3eee5c1"
uuid = "784f63db-0788-585a-bace-daefebcd302b"
version = "8.0.1003+0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

[[deps.Quadmath]]
deps = ["Compat", "Printf", "Random", "Requires"]
git-tree-sha1 = "67fe599f02c3f7be5d97310674cd05429d6f1b42"
uuid = "be4d8f0f-7fa4-5f49-b795-2f01399ab2dd"
version = "0.5.10"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "7f4228017b83c66bd6aa4fddeb170ce487e53bc7"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.6.2"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "3a7c7e5c3f015415637f5debdf8a674aa2c979c4"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.2.1"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.Spglib]]
deps = ["CrystallographyCore", "StaticArrays", "StructEquality", "spglib_jll"]
git-tree-sha1 = "3ba34ed365df96e26e78c192de4448ad1ec4998e"
uuid = "f761d5c5-86db-4880-b97f-9680a7cccfb5"
version = "0.9.4"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools"]
git-tree-sha1 = "87d51a3ee9a4b0d2fe054bdd3fc2436258db2603"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.1.1"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Static"]
git-tree-sha1 = "96381d50f1ce85f2663584c8e886a6ca97e60554"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.8.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "777657803913ffc7e8cc20f0fd04b634f871af8f"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.8"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StructEquality]]
deps = ["Compat"]
git-tree-sha1 = "192a9f1de3cfef80ab1a4ba7b150bb0e11ceedcf"
uuid = "6ec83bb0-ed9f-11e9-3b4c-2b04cb4e219c"
version = "2.1.0"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "3a6f063d690135f5c1ba351412c82bae4d1402bf"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.25"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d95fe458f26209c66a187b1114df96fd70839efd"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.21.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulAtomic]]
deps = ["Unitful"]
git-tree-sha1 = "903be579194534af1c4b4778d1ace676ca042238"
uuid = "a7773ee8-282e-5fa2-be4e-bd808c38a91a"
version = "1.0.0"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "6a451c6f33a176150f315726eba8b92fbfdb9ae7"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.4+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "a54ee957f4c86b526460a720dbc882fa5edcbefc"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.41+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "15e637a697345f6743674f1322beefbc5dcd5cfc"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.6.3+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "326b4fea307b0b39892b3e85fa451692eda8d46c"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.1+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "3796722887072218eabafb494a13c963209754ce"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d2d1a5c49fae4ba39983f63de6afcbea47194e85"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "bcd466676fef0878338c61e655629fa7bbc69d8e"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "555d1076590a6cc2fdee2ef1469451f872d8b41b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "936081b536ae4aa65415d869287d43ef3cb576b2"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.53.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "b70c870239dc3d7bc094eb2d6be9b73d27bef280"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.44+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.spglib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "bc328924cf4975fe49e6416f7e1622e8ceda55e8"
uuid = "ac4a9f1e-bdb2-5204-990c-47c8b2f70d4e"
version = "2.1.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─4b7ade30-8c59-11ee-188d-c3d4588f7106
# ╟─35066ab2-d52d-4a7f-9170-abbae6216996
# ╠═a9548adc-cc71-4f26-96d2-7aefc39a554f
# ╟─f09f563c-e458-480f-92be-355ab4582fb5
# ╠═3de7dfe4-032b-4e05-a929-5ffabfbb35cc
# ╟─c17073d6-94e2-464f-acff-6f444bf7d08c
# ╟─a4f9260b-7016-4a12-8fd1-230b8e234496
# ╟─1bfb62aa-4b01-4839-8073-2a196162cd17
# ╠═d5c66274-47ec-41e3-a539-dc932ac7939a
# ╟─9b6218d6-b621-4ffd-b55a-fe3845357420
# ╠═2689ed80-d936-4044-943b-b198cf57cdd1
# ╟─8c8e8409-4e8a-4e29-875c-dbc969a3634d
# ╠═0e3c162b-bfbc-42e0-986c-aebb328cc2ed
# ╟─347744bb-8670-4d67-855d-982ae6c35f86
# ╠═27df213e-7e3e-4d25-880b-d0cce3563c04
# ╟─50cc1f78-4de0-48f4-876a-d30b4ef3d8d0
# ╠═915c79fc-5d79-4d61-acac-5e4b9618814d
# ╠═cfdbc7d7-21dd-4639-9a55-f9aa11ce746e
# ╠═099a4340-bb0b-4eef-a630-ea94fba4d70c
# ╠═9dd3d49a-ac5a-472e-aec6-d758ab604e08
# ╟─cb07dfa3-5151-4993-a8b4-62190ccace02
# ╠═5432afd8-3e00-4756-ab89-d0315625330d
# ╠═6903d208-e9df-4a3e-b938-95ebee57ea90
# ╠═f32f840c-c8f4-437a-8ac9-5b6cba2abb34
# ╠═86693da0-9b06-4c13-9d73-55ee364de1d8
# ╠═0a6c99c7-11c8-46f4-8ec2-6fb2166579f7
# ╟─3d5e051f-13a9-415e-80bd-78cafc16b97a
# ╠═da1c7f17-aaf8-464a-b7b5-8ce2f2c5a1c4
# ╠═d870beac-018e-44c6-8356-8ba4e337c2cd
# ╟─eab701f2-b43e-4805-8a78-ee62dde5fc15
# ╟─d32943b3-d098-4b9b-bab0-5b1871d9d5ac
# ╠═8d20abfb-63ff-424b-a560-cbad23ad1e84
# ╟─58e69281-e2d7-40d9-9c1e-4b6807b405b7
# ╠═cd99bdc8-dabf-4ddc-82a7-e79cd0f2e52b
# ╠═6002d1e6-57b6-4284-a773-409308e4b58f
# ╟─f7343721-e40b-437d-871e-b7b52201d861
# ╠═89dc90a0-6c2c-4076-86b2-7ed79eccc988
# ╠═8a5692e8-ebf4-4521-9d23-aff2bb0c4921
# ╠═b68f0dce-89a4-4a9c-8f2e-4611895063df
# ╠═03f6cbbf-f68f-40ed-9580-984ffa36f4e2
# ╠═e4a34755-7a66-4bf8-bc90-7751188c7300
# ╠═ba2590e6-c9f6-46ff-84e0-20ecef148338
# ╠═9127d505-3d0b-4ad0-b08e-ce441a74991e
# ╟─772542e0-4676-46dc-827b-51dbeaa7f942
# ╟─ff257f67-2571-408a-88de-6afe220d2a67
# ╠═727cb564-bf00-431a-a4d8-26d5fa2beddc
# ╟─3a0175c8-636e-40cd-963e-f7d9b30452a0
# ╠═193a4535-efdc-4697-b734-dbefbd224b19
# ╠═b2f3187e-c468-493b-8d73-e70fb8459e85
# ╠═1026b80b-2f02-4a2c-ab79-3875e59a6836
# ╠═c9cc14a6-4054-4fde-8b7e-abc0f4f8ffa4
# ╟─bb62070f-4277-4ab5-9a8c-1f865c07aeab
# ╠═254bfb5d-8b0b-4cba-8bc4-c32c0312f690
# ╠═57dcd7d6-edd4-4c02-a3fd-dd74aeb5598a
# ╠═7270fb1f-12eb-4a1a-a578-e3fc068cf698
# ╟─00720b56-b681-4ff2-bc42-4d19318c7f74
# ╠═02ce43cd-9c83-4113-b326-852cf82c1751
# ╠═3652b972-275c-424f-bee1-9f54b7ccd898
# ╠═a7eaade6-323d-4373-96e3-34dc81b3803e
# ╟─e5af416d-20a9-4812-99ff-45d62d6ab3a6
# ╟─66758cad-6aa8-496c-b2a6-3b1ca7c1fe42
# ╠═d6930f90-2ded-4491-be63-582bfdb3b8d8
# ╟─23afdfdd-885d-4aaa-baa0-78948b8fca48
# ╠═376a8e8e-c0be-413f-8035-dea8cd14f752
# ╠═e5bd8031-2c4a-4a62-b869-2ca4da4678ef
# ╟─f2956e16-2e09-4c4e-a663-b8c5acb7b089
# ╠═5b7b3343-1104-4856-b250-0435a3c770f9
# ╠═0575ec37-c8ff-46f1-b8a7-ad1d83b511ca
# ╟─1f5203a7-697e-40de-b5ba-0d44013a0a51
# ╟─776df9aa-f137-41a9-a721-af70fe6439bc
# ╠═87bff49e-d791-482a-94c9-c1486a4ae094
# ╠═4c33a6ca-5be0-44e7-939d-b07c96e0999f
# ╟─3bf2daa3-5920-408a-b32e-0b5c25cd9580
# ╟─0ecd527b-da61-4737-9f05-5f4c851a6077
# ╟─6bdc0066-1f86-458d-aee5-6a7564f339e6
# ╟─d3e0a2e9-9d89-476b-8d37-da7b14904b1e
# ╟─d12f9337-0b87-40cb-817d-8a4ac3858196
# ╟─354b8072-dcf0-4182-9897-c3e9534bef5a
# ╟─9c98cecd-53b9-483e-aecf-c925b5bfd18a
# ╟─ef2796f3-b6d4-4403-aaa6-ed3d9d541a3a
# ╟─7495739f-56b0-4076-8c91-f4155d2ba00f
# ╟─fd4691d1-b46c-444a-96f6-443369634d59
# ╟─3f128396-f3b6-44b0-906e-b1a8467fd43f
# ╟─05d40b5e-fd83-4e73-8c78-dfd986a42fc0
# ╠═c4393902-7c57-4126-80af-8765bea42ebd
# ╠═78cc8d4a-cb63-48d0-a2f9-b8ec8c2950e5
# ╟─0aaeb747-ce4f-4d53-a3b7-f8019c3d688b
# ╠═ffd75a52-9741-4c12-ae2f-303db8d70332
# ╟─0f6a001d-ce82-4d4b-a914-37bf0b62d604
# ╠═66b93963-272f-4e1b-827c-0f3b551ee52b
# ╟─dfe6be2f-fb6b-458b-8016-8d89dfa66ed9
# ╠═252e505d-a43d-472a-9fff-d529c3a2eeb7
# ╟─0f5d7c6c-28d7-4cf9-a279-8117ddf7d9d5
# ╠═b4ffe212-6f71-4d61-989a-570c5448877a
# ╟─5138009b-89e6-4863-b8ec-6ef0e86e2124
# ╠═25775046-d3bc-4e65-80c2-3fa8021b5d86
# ╟─0f3ed8c8-0978-4b7e-a9bd-de902783201a
# ╠═c3a6aec7-ab22-4017-8e2f-bfe5696021dd
# ╟─8f3634ca-9c13-4d40-af3c-0c80588fd8c8
# ╠═ac0cffd2-effb-489c-a118-60864798d55e
# ╟─142ac96e-bd9d-45a7-ad31-e187f28e884a
# ╠═601c837c-1615-4826-938c-b39bb35f46d1
# ╟─e0a07aca-f81a-436b-b11e-8446120e0235
# ╠═56108a52-61ed-43fc-bd0d-7fc4221f05ce
# ╠═580e4dc0-b4dd-4f63-9db7-7fdebf225f1c
# ╠═bfcf40d8-3855-454b-843b-fe3bdd1e5a92
# ╠═20c15f08-8bf9-4c6b-84ab-4486ba0a9237
# ╠═e0c6e8d3-4f0a-4be6-805f-79fb012c01d8
# ╠═0e0ed0c6-ab11-4d4e-9552-3c89f7bd1392
# ╠═5daa306b-0dfe-4d4b-9e03-1ed7b50d6c61
# ╟─ba1ba08f-7fe5-4fb3-b736-adaf7200af08
# ╟─58856ccd-3a1c-4a5f-9bd2-159be331f07c
# ╟─e04087db-9973-4fad-a964-20d109fff335
# ╟─83b6b542-0eb1-4ff7-bea6-26466af478f5
# ╟─abdfcc43-245d-425a-809e-d668f03e9b45
# ╠═ceb72ee7-f7ac-4a49-a585-78e1d55cde2a
# ╟─d26fec73-4416-4a20-bdf3-3a4c8ea533d1
# ╠═b1270539-8079-4761-ab61-55d08d563635
# ╟─9a953b4e-2eab-4cb6-8b12-afe754640e22
# ╟─0616cc6b-c5f8-4d83-a247-849a3d8c5de8
# ╠═d363fa0a-d48b-4904-9337-098bcef015bb
# ╟─9e7d9b0f-f29d-468f-8f7c-e01f271d57c6
# ╠═d34fa8e5-903f-45f4-97f8-32abd8883e9f
# ╠═e3244165-9f04-4676-ae80-57667a6cb8d5
# ╟─968499a6-c0c3-484b-a006-9bbdccca62ba
# ╠═61448d06-c628-493b-8fa0-a8f0a41acd1d
# ╟─722f34dd-e1cb-4a3c-9a13-7ff6117029fc
# ╠═06270fe2-df78-431a-8433-d69fc89ae5c3
# ╟─01c80f21-b18b-4f84-9758-ada4895095f1
# ╠═586e5ae6-f0f8-4222-8a7c-aa687c886685
# ╟─e56b2164-5f66-4745-b1a5-711dcc1a324b
# ╠═66992c0d-262c-4211-b5d2-1ba247f9e8ba
# ╟─86f70a52-13ce-42b5-9a28-2c01d92b022d
# ╠═2dee9a6b-c5dd-4511-a9e3-1d5bc33db946
# ╟─6986bb41-aed6-4ca6-aa70-90cb1fe0bfc3
# ╟─da8c23eb-d473-4c2f-86f9-879016659f3e
# ╠═3006a991-d017-40b3-b52b-31438c05d088
# ╟─646242a2-8df6-4caf-81dc-933345490e6c
# ╠═1957a8bb-b426-49c3-be25-2ad418d292ad
# ╟─d7f3cadf-6161-41ce-8a7a-20fba5182cfb
# ╟─5f0d20d1-c5f0-43b8-b65d-d5a2039cd01a
# ╠═4cc3ded3-9c62-4af5-be73-7f52fa6ed177
# ╟─a1c02b0d-da66-4b1a-9a4b-2a5dd7997232
# ╠═21632ccc-690f-4fed-9091-a9444e0a79d2
# ╟─7ed09f35-a22b-47c0-9ada-e3c11a9645a8
# ╠═63b1a886-6b63-4219-a941-f1a699f0d614
# ╟─5f5f139e-57f9-4b2f-8398-a13fbe119fb5
# ╟─14df1c0f-1c4f-4da9-be49-3941b9c12fd3
# ╟─3fc07beb-33c1-43b3-9d66-27693d78e46a
# ╟─6a8ceca2-05f1-4846-b620-fe44d4c3e0d3
# ╟─012d944c-be57-4021-994f-eb0090b52e2b
# ╟─047d630b-e85e-45e9-9574-758955cb160e
# ╟─57a4ebd3-d850-4490-9992-957c15d26aea
# ╟─677a71e3-b17b-42b2-b3c7-f21cd0f81861
# ╟─a70934fa-e78f-4592-90f8-57af5e9cb87a
# ╟─3fd29e37-d6ea-427a-9b25-2e1dcba42ed0
# ╟─d5de7b64-db19-48cf-81e4-d1f48c6ed08b
# ╟─e2ff4156-5894-4ed6-a52e-6909c5f03bf8
# ╟─5bfb143e-414b-4283-9a6a-be99452aa1b5
# ╟─d6e7e0d0-52d8-4306-bde0-1a55598ed6af
# ╟─d4792fec-7762-4c2b-9be8-3c6c6791bd3d
# ╠═10426573-4929-4576-9633-781b024bd97b
# ╟─3de33cb8-c6ee-4b60-94df-2cf29afbe2d7
# ╟─2c6f639e-fbf2-4d9a-a0dd-4306f0b92a8d
# ╠═f0b9d21b-dc78-4d95-ac47-5a19f41c0fd4
# ╟─6a83edeb-fe78-4ad5-98a2-d1ef33c333e4
# ╟─33896109-d190-4992-806a-c447ca36071b
# ╠═93552ccd-f7b5-4830-a9ad-417d3fca9af9
# ╟─968a26f2-12fe-446f-aa41-977fdffc23a0
# ╠═7ebc18b0-618f-4d99-881f-7c30eb3bc7f5
# ╟─bf00ec9d-5c69-43c3-87c1-28ac9f5b4c9f
# ╟─9430a871-0f8f-4ee2-b574-f818f1580abd
# ╠═3c1c0131-96fe-4fb7-ac0c-0aaec21dd023
# ╟─5de67497-d252-443a-8a72-8e01891aea41
# ╠═8c712ee5-f076-442e-ad33-edfe5a6e6941
# ╟─ee3fe129-48f8-4932-ac23-2287c6be9780
# ╟─b61a9e8c-e7b4-4e5e-8b4a-10e939c164a9
# ╟─ea9b6c1b-e2dd-4c68-82ca-b63ada51cfd1
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
