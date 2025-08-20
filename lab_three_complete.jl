# %% [markdown]
# # Lab 3: A Conditional Generative Model for Images
# Welcome to lab 3! In the previous lab, we studied *unconditional* generation, for toy, two-dimensional data distributions. In this lab, we will study *conditional* generation on *images* from the MNIST dataset of handwritten digits. Each such MNIST image is not two dimensions but $32\times 32 = 1024$ dimensions! The nature of our new, more challenging setting will require us to take special care:
# 1. To tackle *conditional* generation, we will employ *classifier-free guidance* (CFG) (see Part 2.1).
# 2. To parameterize our learned vector field for high-dimensional image-valued data, a simple MLP will not suffice. Instead, we will adopt the *U-Net* architecture (see part 2.2).
#
# If you find any mistakes, or have any other feedback, please feel free to email us at `erives@mit.edu` and `phold@mit.edu`. Enjoy!

using Adapt
using CairoMakie
using Distributions
using ForwardDiff
using LinearAlgebra
using Lux
using LuxCUDA
using NNlib
using Optimisers
using Random
# using WGLMakie
using Zygote
using MLDatasets
using MLUtils

outdir = joinpath(@__DIR__, "output", "lab_three") |> mkpath

function loadmnist(batchsize, split)
    dataset = MNIST(; split)
    imgs = dataset.features
    labels_raw = dataset.targets

    # Process images into (H,W,C,BS) batches
    y = labels_raw .+ 1 |> collect
    z = reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3)) |> f32 |> collect

    # Normalize
    μ, σ = 5f-1, 5f-1
    @. z = (z - μ) / σ

    # Use DataLoader to automatically minibatch and shuffle the data
    DataLoader((y, z); batchsize, shuffle = true, partial = false)
end

silu(x) = @. x / (1 + exp(-x))

rng = Random.Xoshiro(0)

data_test = MNIST(:test)

data_train.metadata
data_train.features |> size
data_train.targets
data_train.split

function implot!(ax, x; kwargs...)
    x = reverse(x; dims = 2)
    image!(ax, x; kwargs...)
end
function implot(x)
    fig = Figure()
    ax = Axis(fig[1, 1]; aspect = DataAspect())
    implot!(ax, x)
    fig
end

let
    fig = Figure(; size = (500, 500))
    n = 5
    for i = 1:n, j = 1:n
        ax = Axis(
            fig[i, j];
            aspect = DataAspect(),
            xticksvisible = false,
            yticksvisible = false,
            xticklabelsvisible = false,
            yticklabelsvisible = false,
        )
        implot!(ax, data_train[i+n*(j-1)].features)
    end
    fig
end

data_train[1].targets

# %% [markdown]
# # Part 3: An Architecture for Images
# At this point, we have discussed classifier free guidance, and the necessary considerations that must be made on the part of our model and in training our model. What remains is to actually discuss the choice of model. In particular, our usual choice of an MLP, while fine for the simple distributions of the previous lab, will no longer suffice. To this end, we will a new convolutional architecture - the **U-Net** - which is specifically tailored toward images. A diagram of the U-Net we'll be using is shown below. ![image.png](attachment:bd703834-9239-4ed3-b8c1-9639fc971575.png)

# %% [markdown]
# ### Question 3.1: Building a U-Net

# %% [markdown]
# Below, we implement the U-Net shown in the diagram above.

# %%
# Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
function FourierEncoder(dim)
    @assert dim % 2 == 0
    half_dim = div(dim, 2)
    weights = randn(Float32, half_dim) |> gpu_device()
    @compact(; weights,) do t
        t = reshape(t, 1, :)
        freqs = @. 2 * t * weights
        sin_embed = @. sqrt(2.0f0) * sinpi(freqs)
        cos_embed = @. sqrt(2.0f0) * cospi(freqs)
        output = vcat(sin_embed, cos_embed)
        @return output
    end
end

ResidualLayer(channels, time_embed_dim, y_embed_dim) =
    @compact(;
        block1 = Chain(
            silu,
            BatchNorm(channels),
            Conv((3, 3), channels => channels; pad = 1),
        ),
        block2 = Chain(
            silu,
            BatchNorm(channels),
            Conv((3, 3), channels => channels; pad = 1),
        ),
        time_adapter = Chain(
            Dense(time_embed_dim => time_embed_dim, silu),
            Dense(time_embed_dim => channels),
        ),
        y_adapter = Chain(
            Dense(y_embed_dim => y_embed_dim, silu),
            Dense(y_embed_dim => channels),
        ),
    ) do (x, t_embed, y_embed)
        res = copy(x)

        # Initial conv block
        x = block1(x)

        # Add time embedding
        t_embed = time_adapter(t_embed)
        t_embed = reshape(t_embed, 1, 1, size(t_embed)...)
        x = x .+ t_embed

        # Add y embedding (conditional embedding)
        y_embed = y_adapter(y_embed)
        y_embed = reshape(y_embed, 1, 1, size(y_embed)...)
        x = x .+ y_embed

        # Second conv block
        x = block2(x)

        # Add back residual
        x = x .+ res

        @return x
    end

Encoder(channels_in, channels_out, num_residual_layers, t_embed_dim, y_embed_dim) =
    @compact(;
        res_blocks = fill(
            ResidualLayer(channels_in, t_embed_dim, y_embed_dim),
            num_residual_layers,
        ),
        downsample = Conv((3, 3), channels_in => channels_out; stride = 2, pad = 1),
    ) do (x, t_embed, y_embed)
        for block in res_blocks
            x = block((x, t_embed, y_embed))
        end
        x = downsample(x)
        @return x
    end

let
    x = randn(Float32, 28, 28, 32, 3)
    t_embed = randn(Float32, 40, 3)
    y_embed = randn(Float32, 40, 3)
    # net = ResidualLayer(32, 40, 40)
    net = Encoder(32, 64, 2, 40, 40)
    ps, st = Lux.setup(rng, net)
    net((x, t_embed, y_embed), ps, Lux.testmode(st)) |> first |> size
end

Midcoder(channels, num_residual_layers, t_embed_dim, y_embed_dim) =
    @compact(;
        res_blocks = fill(
            ResidualLayer(channels, t_embed_dim, y_embed_dim),
            num_residual_layers,
        ),
    ) do (x, t_embed, y_embed)
        for block in res_blocks
            x = block((x, t_embed, y_embed))
        end
        @return x
    end

Decoder(channels_in, channels_out, num_residual_layers, t_embed_dim, y_embed_dim) =
    @compact(;
        upsample = Chain(
            Upsample(:bilinear; scale = 2),
            Conv((3, 3), channels_in => channels_out; pad = 1),
        ),
        res_blocks = fill(
            ResidualLayer(channels_out, t_embed_dim, y_embed_dim),
            num_residual_layers,
        ),
    ) do (x, t_embed, y_embed)
        x = upsample(x)
        for block in res_blocks
            x = block((x, t_embed, y_embed))
        end
        @return x
    end

MNISTUNet(; channels, num_residual_layers, t_embed_dim, y_embed_dim) =
    @compact(;
        # Initial convolution: (bs, 1, 32, 32) -> (bs, c_0, 32, 32)
        init_conv = Chain(
            Conv((3, 3), 1 => channels[1]; pad = 1),
            BatchNorm(channels[1]),
            silu,
        ),

        # Initialize time embedder
        time_embedder = FourierEncoder(t_embed_dim),

        # Initialize y embedder
        y_embedder = Embedding(11 => y_embed_dim),

        # Encoders, Midcoders, and Decoders
        encoders = map(
            i -> Encoder(
                channels[i],
                channels[i+1],
                num_residual_layers,
                t_embed_dim,
                y_embed_dim,
            ),
            1:length(channels)-1,
        ),
        decoders = map(
            i -> Decoder(
                channels[i],
                channels[i-1],
                num_residual_layers,
                t_embed_dim,
                y_embed_dim,
            ),
            length(channels):-1:2,
        ),
        midcoder = Midcoder(channels[end], num_residual_layers, t_embed_dim, y_embed_dim),

        # Final convolution
        final_conv = Conv((3, 3), channels[1] => 1; pad = 1),
    ) do (x, t, y)
        # Embed t and y
        t_embed = time_embedder(t)
        y_embed = y_embedder(y)

        # Initial convolution
        x = init_conv(x) # (bs, c_0, 32, 32)

        residuals = ()

        # Encoders
        for encoder in encoders
            x = encoder((x, t_embed, y_embed)) # 2w x 2h x c_i x bs -> w x h x c_{i+1} x bs
            residuals = residuals..., copy(x)
        end

        # Midcoder
        x = midcoder((x, t_embed, y_embed))

        # Decoders
        for decoder in decoders
            residuals..., res = residuals
            x = x + res
            x = decoder((x, t_embed, y_embed)) # w x h x c_i x bs -> 2w x 2h x c_{i-1} x bs
        end

        # Final convolution
        x = final_conv(x) # 32 x 32 x 1 x bs

        @return x
    end

# %% [markdown]
# **Your job**: Pick *two* components of the architecture above (each one of `FourierEncoder`, `ResidualLayer`, `Encoder`, `Decoder`, or `Midcoder`), and explain, in your own words, (1) their role in the U-Net, (2) their inputs and outputs, and (3) a brief description of how the inputs turn into outputs.
#
# **Your answer**:

# %% [markdown]
# ### Question 3.2: Training a U-Net for Classifier-Free Guidance

# %% [markdown]
# Now let's train!

# %%
# Initialize model
unet = let
    device = gpu_device()
    model = MNISTUNet(;
        channels = [32, 64, 128],
        num_residual_layers = 2,
        t_embed_dim = 40,
        y_embed_dim = 40,
    )
    ps, st = Lux.setup(rng, model) |> device
    train_state = Training.TrainState(model, ps, st, Adam(1.0f-3))
    nepoch = 20
    nsample = 250
    eta = 0.1
    data_train = loadmnist(nsample, :train)
    loss = MSELoss()
    for iepoch = 1:nepoch, (ibatch, batch) in enumerate(data_train)
        y, z = batch |> device
        # Set each label to 10 (i.e., null) with probability eta
        @. y = ifelse(rand() > eta, y, 11)
        x0 = randn!(similar(z))
        t = rand!(similar(z, nsample))
        t .*= 0.999f0 # Definition of target `u` is not good for `t` close to 1
        tre = reshape(t, 1, 1, 1, :)
        x = @. tre * z + (1 - tre) * x0
        u = @. (z - x) / (1 - tre) # Linear conditional vector field
        _, l, _, train_state =
            Training.single_train_step!(AutoZygote(), loss, ((x, t, y), u), train_state)
        ibatch % 1 == 0 && @info "iepoch = $iepoch, ibatch = $ibatch, loss = $l"
    end
    ps_freeze = train_state.parameters
    st_freeze = train_state.states
    (x, t, y) -> first(model((x, t, y), ps_freeze, Lux.testmode(st_freeze)))
end

# %% [markdown]
# How well does our model do? Let's find out! We'll use the class `CFGVectorFieldODE` to wrap the UNet in an instance of `ode` so that we can integrate it!

# %%
let
    # Play with these!
    nsample = 10
    nstep = 100
    guidance_scales = [0f0, 3f0, 5f0]
    fig = Figure(; size = (800, 300))
    for (i, w) in enumerate(guidance_scales)
        labels = 1:11
        nlabel = length(labels)
        y = stack(fill(labels, nsample); dims = 1) |> vec |> gpu_device()
        x = randn(Float32, 28, 28, 1, nsample)
        x = reshape(stack(fill(x, nlabel)), 28, 28, 1, :) |> gpu_device()
        t = 0.0f0
        for j = 1:nstep
            @show t
            h = 1.0f0 / nstep
            tcast = fill(t, nsample * nlabel) |> gpu_device()
            uy = unet(x, tcast, y)
            ynull = fill!(copy(y), 11)
            unull = unet(x, tcast, ynull)
            u = (1 - w) * unull + w * uy
            @. x += h * u
            t += h
        end
        ax = Axis(
            fig[1, i];
            title = "Guidance: w = $w",
            xticksvisible = false,
            yticksvisible = false,
            xticklabelsvisible = false,
            yticklabelsvisible = false,
            aspect = DataAspect(),
        )
        x = x |> cpu_device()
        x = reverse(x; dims = 2) # Reorient for plotting
        x = reshape(x, 28, 28, nsample, nlabel)
        x = permutedims(x, (1, 3, 2, 4))
        x = reshape(x, 28 * nsample, 28 * nlabel)
        image!(ax, x; interpolate = false, colorrange = (-1, 1))
    end
    file = "$outdir/results.pdf"
    @info "Saving to $file"
    save(file, fig; backend = CairoMakie)
    fig
end
