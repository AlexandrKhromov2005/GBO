# GBO Watermark Embedding Project

This project implements a Genetic Bee Optimization (GBO)-based algorithm for robust grayscale image watermarking using 8×8 DCT blocks.  
It is written in modern C++ (C++17) and relies on Armadillo, OpenCV, and GoogleTest.

---

## 1. Prerequisites

| Library | Minimum Version | Ubuntu/Debian package |
|---------|-----------------|-----------------------|
| CMake   | 3.16            | `cmake` |
| GCC / Clang | C++17 capable | `build-essential` or `clang` |
| Armadillo | 9.800          | `libarmadillo-dev` |
| OpenCV   | 4.x (built with `opencv_contrib` is **not** required) | `libopencv-dev` |
| GoogleTest | 1.11 (or distro-provided) | `libgtest-dev` + `cmake` |

> **Note**  On Ubuntu, `libgtest-dev` ships only the sources. The build script below compiles it automatically, so there is no additional manual step.

### Installing with apt (Ubuntu ≥22.04)
```bash
sudo apt update && sudo apt install -y \
    build-essential cmake git libarmadillo-dev libopencv-dev libgtest-dev
```

If you prefer `clang`, simply `sudo apt install clang` and pass `-DCMAKE_CXX_COMPILER=clang++` during CMake configuration.

---

## 2. Building the project

```bash
# Clone
git clone https://github.com/AlexandrKhromov2005/GBO.git
cd GBO

# Create a separate build directory
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### Running the main program
The executable `main` is generated inside `build/`. Provide the paths to the cover image and binary watermark in `src/main.cpp`, or modify them on the command line:

```bash
./build/main    # if hard-coded paths suit you
```

---

## 3. Running unit tests

```bash
cmake --build build --target unit_tests -j$(nproc)
ctest --test-dir build --output-on-failure
```

All core components (zig-zag conversion, population handling, PSNR calculation, etc.) are covered by GoogleTest.

---

## 4. Project structure (high-level)

```
GBO/
├── include/               # Public headers
├── src/                   # Implementation
├── tests/                 # GoogleTest unit tests
├── images/                # Sample images & watermarks
├── CMakeLists.txt         # Top-level build script
└── README.md              # This file
```

---

## 5. Customizing paths
The helper `launchGBO` now receives **all** file paths as parameters (image, watermark, output images). Only `main.cpp` hard-codes default paths; feel free to pass alternatives through CLI arguments or adapt the file.

---

## 6. Troubleshooting
1. **Missing Armadillo / OpenCV headers** – ensure the dev packages (`-dev`) are installed.
2. **Undefined references when linking GoogleTest** – verify that CMake found GTest (look for `Found GTest` messages). If not, build and install GoogleTest manually:
   ```bash
   cd /usr/src/googletest && sudo cmake . && sudo make install
   ```
3. **Image paths invalid** – ensure the `images/` directory exists and contains the specified files.

---

## 7. Customizing the dataset

### Adding new images
Add image file names to `include/dataset_builder.h` inside the `images` vector:
```cpp
const std::vector<std::string> images = {
    "images/airplane.png",
    // ... existing entries ...
    "images/your_new_image.png"
};
```
Put the actual image files in the `images/` directory so the paths remain valid.

### Adding or removing attacks
Edit `src/dataset_builder.cpp` and adjust the constant `attacks` vector located near the top of the file:
```cpp
const std::vector<AttackType> attacks = {
    AttackType::ContrastIncrease,
    AttackType::JPEGCompression,
    // AttackType::YourNewAttack,
};
```

