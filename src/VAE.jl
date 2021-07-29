export VAE, encode, reconstruct, calculate_loss, train!

using Flux
using Flux: @functor
using Setfield: @set
using Flux.Losses: logitbinarycrossentropy
using Random
using Statistics
using Distributions: Normal

"""
"""

mutable struct VAE{
    EM,
    DM,
    O
}
    encoder::EM
    decoder::DM
    latent_dims::Int
    optimizer::O
end

function VAE(;
    encoder::EM,
    decoder::DM,
    latent_dims::Int,
    optimizer::O
) where {EM, DM, O}
    VAE(
        encoder,
        decoder,
        latent_dims,
        optimizer
    )
end

Flux.functor(x::VAE) = (em = x.encoder, dm = x.decoder),
y -> begin
    x = @set x.encoder = y.em
    x = @set x.decoder = y.dm
    x
end

function encode(model::VAE, x::Array{T}) where T
    latent_dims = model.latent_dims
    result = model.encoder(x)
    μ = result[1:latent_dims, :]
    logσ = result[latent_dims+1:2*latent_dims, :]
    return μ, logσ
end

function reconstruct(model::VAE, x::Array{T}) where T
    μ, logσ = encode(model, x)
    z = reparamaterize.(μ, logσ)
    x′ = model.decoder(z)
    return μ, logσ, x′
end

function reparamaterize(μ::T, logσ::T) where {T}
    return Float32.(rand(Normal(0, 1))) * exp(logσ * 0.5f0) + μ
end

function calculate_loss(model::VAE, x::Array{T}) where {T}
    μ, logσ, x′ = reconstruct(model, x)
    batch_size = size(x)[end]

    kl_loss = mean(0.5f0 * sum(-1.0f0 .+ μ.^2 .+ exp.(2.0f0 .* logσ) .- 2.0f0 .* logσ))

    reconstruct_loss = -logitbinarycrossentropy(x, x′)
    kl_loss - reconstruct_loss
end

function train!(model::VAE, x::Array{T}) where {T}
    ps = Flux.params(model)
    loss, back = Flux.pullback(ps) do
        calculate_loss(model, x)
    end
    println(loss)
    grad = back(1f0)
    Flux.Optimise.update!(model.optimizer, ps, grad)
end