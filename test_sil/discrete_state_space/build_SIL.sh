rm -rf ./test_sil/discrete_state_space/build
mkdir -p ./test_sil/discrete_state_space/build
cmake -S ./test_sil/discrete_state_space -B ./test_sil/discrete_state_space/build
make -C ./test_sil/discrete_state_space/build

mv ./test_sil/discrete_state_space/build/DiscreteStateSpaceSIL.*so ./test_sil/discrete_state_space/
