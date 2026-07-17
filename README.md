# Dennis Nduto

**ML Systems & Software Engineer**

I build test-driven software focused on automatic differentiation,
graph compilation, distributed runtimes, numerical correctness, and
production engineering.

My current portfolio covers:

- JAX custom transformations and implicit differentiation
- Custom primitives, batching rules, and StableHLO/MLIR lowering
- Explicit array sharding, device meshes, and collectives
- Pallas kernel programming and runtime validation
- TensorFlow tracing, distributed training, custom-op, and compiler work
  currently under development
- Python, TypeScript, APIs, databases, testing, and technical documentation

## Selected ML Systems Projects

| Project | Technical focus |
|---|---|
| [JAX StableHLO Custom Primitive](https://github.com/dennisnduto/jax-stablehlo-custom-primitive) | Abstract evaluation, custom JVP, batching rules, and MLIR lowering |
| [JAX Mesh Sharding Runtime](https://github.com/dennisnduto/jax-mesh-sharding-runtime) | jax.Array, Mesh, PartitionSpec, shard_map, collectives, and soak testing |
| [JAX Transform Internals Lab](https://github.com/dennisnduto/jax-transform-internals-lab) | custom_jvp, custom_vjp, implicit differentiation, JAXPR, and StableHLO |
| [JAX Pallas Kernel Lab](https://github.com/dennisnduto/jax-pallas-kernel-lab) | BlockSpec tiling, fused kernels, correctness tests, and accelerator benchmarking |

## Engineering Standards

My repositories aim to include:

- Reproducible installation
- Automated tests and continuous integration
- Numerical reference comparisons
- Runtime and failure-path validation
- Architecture and design documentation
- Honest hardware and performance reporting

## Current Direction

I am extending this work into TensorFlow internals, including:

- `tf.function` tracing and AutoGraph behavior
- TensorFlow custom C++ and CUDA operations
- `tf.distribute` and collective communication
- Grappler and XLA graph inspection
- SavedModel and serving-runtime validation

## Contact

- Email: dennisnduto418@gmail.com
- LinkedIn: [Dennis Nduto](https://www.linkedin.com/in/dennis-nduto-42a256250)
- Location: Nairobi, Kenya
- Availability: Remote and asynchronous technical work
