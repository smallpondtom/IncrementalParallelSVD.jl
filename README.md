# Incremental Parallel SVD in Julia

This is a Julia implementation of a parallelized (MPI) version of Brand's 
incremental singular value decomposition (iSVD) algorithm. This work converted 
the C++ by [libROM](https://www.librom.net/) to Julia.

## References

- [1] libROM: [https://www.librom.net/](https://www.librom.net/) 
- [2] Brand's iSVD algorithm: [https://www.merl.com/publications/docs/TR2002-24.pdf](https://www.merl.com/publications/docs/TR2002-24.pdf)