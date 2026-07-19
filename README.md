# Dennis Nduto

### ML Systems & Full-Stack Software Engineer

I build reliable software across machine-learning runtimes, compiler tooling, distributed execution, and production web applications. My work ranges from TensorFlow and JAX internals to TypeScript/Next.js and PHP/Laravel systems backed by relational databases.

Based in **Nairobi, Kenya** and open to remote engineering opportunities.

## What I work on

- **ML systems and runtimes:** distributed training, synchronous collectives, explicit sharding, checkpoint recovery, tracing, graph inspection, and runtime diagnostics
- **Compiler and autodiff tooling:** JAX transformations, custom JVP/VJP rules, custom primitives, JAXPR, StableHLO/MLIR lowering, XLA concepts, and Pallas kernels
- **Reliability engineering:** deterministic workloads, numerical validation, subprocess supervision, heartbeats, watchdogs, structured reports, soak tests, and failure-path testing
- **Full-stack applications:** authenticated products, role-based access, AI-assisted workflows, dashboards, APIs, subscriptions, reporting, and database-backed systems
- **Engineering quality:** automated tests, continuous integration, packaging, Docker, reproducible setup, and architecture documentation

## Languages and technologies

| Area | Technologies |
|---|---|
| Languages | Python, TypeScript, JavaScript, PHP, SQL, HTML, CSS, Shell |
| ML systems | TensorFlow, JAX, NumPy, tf.distribute, StableHLO, MLIR, XLA, Pallas |
| Web engineering | Next.js, React, Laravel, Blade, Tailwind CSS, Alpine.js |
| Data and backend | PostgreSQL, MySQL, Prisma, Auth.js / NextAuth, REST APIs |
| Reliability and tooling | pytest, Ruff, GitHub Actions, Docker, Compose, Make, structured logging |
| Runtime topics | Distributed collectives, mesh sharding, autodiff, graph tracing, checkpointing, process supervision |

> GitHub also classifies Blade templates, Dockerfiles, and Makefiles across my repositories; I use them as part of Laravel delivery, containerization, and build automation.

## Featured ML systems projects

| Project | Focus |
|---|---|
| [TensorFlow Distributed Collectives Runtime](https://github.com/dennisnduto/tensorflow-distributed-collectives-runtime) | Multi-process `MultiWorkerMirroredStrategy`, collective validation, deterministic sharding, heartbeats, watchdogs, checkpoint restart, and soak testing |
| [TensorFlow Function Trace Inspector](https://github.com/dennisnduto/tensorflow-function-trace-inspector) | `tf.function` specialization, `ConcreteFunction`, captures, AutoGraph, GraphDef analysis, SavedModel signatures, and numerical parity |
| [JAX Mesh Sharding Runtime](https://github.com/dennisnduto/jax-mesh-sharding-runtime) | `jax.Array`, device meshes, `PartitionSpec`, `shard_map`, collectives, compiled training, and runtime checks |
| [JAX StableHLO Custom Primitive](https://github.com/dennisnduto/jax-stablehlo-custom-primitive) | Abstract evaluation, custom autodiff, batching rules, and portable StableHLO/MLIR lowering |
| [JAX Transform Internals Lab](https://github.com/dennisnduto/jax-transform-internals-lab) | Custom JVP/VJP rules, implicit differentiation, transformation composition, JAXPR, and numerical validation |
| [JAX Pallas Kernel Lab](https://github.com/dennisnduto/jax-pallas-kernel-lab) | BlockSpec tiling, fused kernels, CPU interpret-mode correctness tests, and accelerator-ready benchmarking |

## Application engineering projects

| Project | Focus |
|---|---|
| [StudyMind AI](https://github.com/dennisnduto/StudyMind) | Next.js, React, TypeScript, Prisma, PostgreSQL, Auth.js, grounded document chat, quizzes, analytics, and subscription access |
| [Smart University Timetable](https://github.com/dennisnduto/smartsystem) | Laravel, PHP, Blade, MySQL, role-oriented workflows, room tracking, timetable management, reporting, and AI-assistant scaffolding |
| [Room Availability System](https://github.com/dennisnduto/Room-Availability-System) | PHP/MySQL room allocation, authentication, administration, availability dashboards, and reports |
| [Portfolio](https://github.com/dennisnduto/portfolio) | Personal web presence built with HTML |

## Engineering principles

- Prefer observed runtime evidence over unsupported claims
- Test numerical behavior against clear references
- Keep distributed workers in isolated operating-system processes
- Make failures visible through structured logs and reports
- Document hardware and validation limits honestly
- Build software that can be installed, tested, and maintained by another engineer

## Current focus

I am deepening my work in distributed ML runtimes, graph compilers, accelerator programming, and production-grade AI application infrastructure—especially the boundary between high-level frameworks and the systems that execute them reliably.

## Contact

- **Email:** [dennisnduto418@gmail.com](mailto:dennisnduto418@gmail.com)
- **LinkedIn:** [Dennis Nduto](https://www.linkedin.com/in/dennis-nduto-42a256250)
- **GitHub:** [@dennisnduto](https://github.com/dennisnduto)
- **Location:** Nairobi, Kenya
