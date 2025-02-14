# Basic Standalone libcudf C++ application

This C++ example demonstrates a parquet_io libcudf use case and provides a minimal
example of building your own application based on libcudf using CMake.

The example source code loads a parquet file and output another parquet file

## Compile and execute

```bash
# Configure project
cmake -S . -B build/
# Build
cmake --build build/ --parallel $PARALLEL_LEVEL
# Execute
build/parquet_io
```

If your machine does not come with a pre-built libcudf binary, expect the
first build to take some time, as it would build libcudf on the host machine.
It may be sped up by configuring the proper `PARALLEL_LEVEL` number.
