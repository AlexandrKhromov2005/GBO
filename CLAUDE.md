# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

### Primary Build
```bash
# Build with CMake (recommended)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Or use Makefile wrapper
make all
```

### Debug Build with Verbose Logging
```bash
# Enable debug logging at compile time
cmake -B build -S . -DENABLE_DEBUG_LOG=ON
# Or with Makefile
make DEBUG=1
```

### Testing
```bash
# Run all tests
cmake --build build --target unit_tests -j$(nproc)
ctest --test-dir build --output-on-failure

# Or with Makefile
make test
```

### Dataset Operations
```bash
# Build dataset for evaluation
./build/main --build-dataset
# Or with Makefile
make build_dataset

# Clear dataset directory
make clear_dataset

# Clean generated images (keep originals)
make rm_images
```

## High-Level Architecture

This is a C++ implementation of a **Genetic Bee Optimization (GBO) algorithm for robust grayscale image watermarking** using 8×8 DCT blocks.

### Core Algorithm Flow
1. **Image Processing**: Split input images into 8×8 blocks (`process_images.h/cpp`)
2. **DCT & Zig-zag**: Convert blocks to frequency domain and apply zig-zag ordering (`process_block.h/cpp`)
3. **Population Management**: Initialize and evolve candidate solutions (`population.h/cpp`)
4. **GBO Optimization**: Main optimization loop with bee behavior simulation (`gbo.h/cpp`)
5. **Watermark Embedding/Extraction**: Embed/extract bits in DCT coefficients (`launch.h/cpp`)
6. **Robustness Testing**: Apply various attacks and measure quality metrics (`attacks.h/cpp`, `metrics.h/cpp`)

### Key Components

#### GBO Algorithm (`include/gbo.h`, `src/gbo.cpp`)
- Main optimization engine with 40 iterations by default
- Implements bee colony behavior (exploration/exploitation phases)
- Uses fitness function based on imperceptibility and robustness

#### Population Management (`include/population.h`, `src/population.cpp`) 
- Manages 30 individuals (candidate solutions) by default
- Each individual represents modification vector for DCT coefficients
- Tracks fitness values and best/worst individuals

#### Block Processing (`include/process_block.h`, `src/process_block.cpp`)
- DCT transformation and zig-zag reordering of 8×8 blocks
- Two embedding schemes with different coefficient regions:
  - Scheme 0: Basic s1/s0 region mapping
  - Scheme 1: Extended region mapping with additional coefficients
- Fitness calculation based on PSNR and bit extraction reliability

#### Attack Simulation (`include/attacks.h`, `src/attacks.cpp`)
- 12 different attack types: brightness/contrast changes, noise, filtering, JPEG compression
- Used for robustness evaluation during optimization and testing

#### Dataset Builder (`include/dataset_builder.h`, `src/dataset_builder.cpp`)
- Automated dataset generation with 8 test images
- Applies all attack combinations for comprehensive evaluation
- Can embed uniform bits (0 or 1) across entire images

### Main Entry Points

#### Single Run Mode (default)
```cpp
launchGBO(image_path, watermark_path, scheme);
```

#### Multi-trial Evaluation Mode
```bash
./build/main --trials N
```
Runs N trials and aggregates metrics (BER, PSNR, SSIM, NCC, MSE) across all attacks.

#### Dataset Building Mode  
```bash
./build/main --build-dataset
```

### File Structure Patterns
- **Headers**: `include/` - Public interfaces and constants
- **Implementation**: `src/` - Algorithm implementations  
- **Tests**: `tests/` - GoogleTest unit tests covering core functions
- **Images**: `images/` - Test images and watermarks
- **Dataset**: `dataset/` - Generated evaluation datasets

### Dependencies
- **Armadillo**: Matrix operations and linear algebra
- **OpenCV**: Image I/O and basic processing
- **GoogleTest**: Unit testing framework

### Debug Mode
When `ENABLE_DEBUG_LOG` is defined, detailed logging is available throughout the optimization process. Use `DEBUG=1` with make or `-DENABLE_DEBUG_LOG=ON` with cmake.