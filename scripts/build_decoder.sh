cd ../my_utils/decoder/cbox_decode
rm -rf build cbox_decode.so
python3 build.py build_ext --inplace
cd -

cd ../my_utils/decoder/cpose_decode
rm -rf build cpose_decode.so
python3 build.py build_ext --inplace
cd -

cd ../my_utils/decoder/cseg_decode
rm -rf build cseg_decode.so
python3 build.py build_ext --inplace
cd -

cd ../my_utils/decoder/tracking/cbytetrack
rm -rf build
mkdir build
cd -

cd ../my_utils/decoder/tracking/cbytetrack/build
cmake ..
make
cd -