cmake -S . -B build -DRABITQ_BUILD_EXPERIMENTS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target run_experiment -j 2
./build/sigmod_experiments/run_experiment mf_dino2